я€:
Чи
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8јщ8
z
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/kernel
s
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*"
_output_shapes
: *
dtype0
n
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d/bias
g
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes
: *
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
: @*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:@*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
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
lstm_3/lstm_cell_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А**
shared_namelstm_3/lstm_cell_3/kernel
И
-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/kernel*
_output_shapes
:	@А*
dtype0
£
#lstm_3/lstm_cell_3/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*4
shared_name%#lstm_3/lstm_cell_3/recurrent_kernel
Ь
7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_3/lstm_cell_3/recurrent_kernel*
_output_shapes
:	 А*
dtype0
З
lstm_3/lstm_cell_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_3/lstm_cell_3/bias
А
+lstm_3/lstm_cell_3/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell_3/bias*
_output_shapes	
:А*
dtype0
П
lstm_4/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А**
shared_namelstm_4/lstm_cell_4/kernel
И
-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/kernel*
_output_shapes
:	 А*
dtype0
£
#lstm_4/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*4
shared_name%#lstm_4/lstm_cell_4/recurrent_kernel
Ь
7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_4/recurrent_kernel*
_output_shapes
:	@А*
dtype0
З
lstm_4/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namelstm_4/lstm_cell_4/bias
А
+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/bias*
_output_shapes	
:А*
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
Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d/kernel/m
Б
(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*"
_output_shapes
: *
dtype0
|
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv1d/bias/m
u
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes
: *
dtype0
М
Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv1d_1/kernel/m
Е
*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
: @*
dtype0
А
Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
Э
 Adam/lstm_3/lstm_cell_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/m
Ц
4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/m*
_output_shapes
:	@А*
dtype0
±
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
™
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m*
_output_shapes
:	 А*
dtype0
Х
Adam/lstm_3/lstm_cell_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_3/lstm_cell_3/bias/m
О
2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/m*
_output_shapes	
:А*
dtype0
Э
 Adam/lstm_4/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/m
Ц
4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/m*
_output_shapes
:	 А*
dtype0
±
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
™
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm_4/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_4/lstm_cell_4/bias/m
О
2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d/kernel/v
Б
(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*"
_output_shapes
: *
dtype0
|
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv1d/bias/v
u
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes
: *
dtype0
М
Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*'
shared_nameAdam/conv1d_1/kernel/v
Е
*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
: @*
dtype0
А
Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0
Э
 Adam/lstm_3/lstm_cell_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/lstm_3/lstm_cell_3/kernel/v
Ц
4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_3/lstm_cell_3/kernel/v*
_output_shapes
:	@А*
dtype0
±
*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*;
shared_name,*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
™
>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v*
_output_shapes
:	 А*
dtype0
Х
Adam/lstm_3/lstm_cell_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_3/lstm_cell_3/bias/v
О
2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_3/lstm_cell_3/bias/v*
_output_shapes	
:А*
dtype0
Э
 Adam/lstm_4/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/v
Ц
4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/v*
_output_shapes
:	 А*
dtype0
±
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
™
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v*
_output_shapes
:	@А*
dtype0
Х
Adam/lstm_4/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/lstm_4/lstm_cell_4/bias/v
О
2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
ГP
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЊO
valueіOB±O B™O
х
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell
 
state_spec
!regularization_losses
"trainable_variables
#	variables
$	keras_api
l
%cell
&
state_spec
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
Ў
;iter

<beta_1

=beta_2
	>decay
?learning_ratemОmПmРmС+mТ,mУ1mФ2mХ@mЦAmЧBmШCmЩDmЪEmЫvЬvЭvЮvЯ+v†,v°1vҐ2v£@v§Av•Bv¶CvІDv®Ev©
 
f
0
1
2
3
@4
A5
B6
C7
D8
E9
+10
,11
112
213
f
0
1
2
3
@4
A5
B6
C7
D8
E9
+10
,11
112
213
≠
Flayer_regularization_losses

regularization_losses
Gmetrics
Hlayer_metrics
trainable_variables

Ilayers
	variables
Jnon_trainable_variables
 
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
Klayer_regularization_losses
regularization_losses
Lmetrics
	variables
Mlayer_metrics
trainable_variables

Nlayers
Onon_trainable_variables
[Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
Player_regularization_losses
regularization_losses
Qmetrics
	variables
Rlayer_metrics
trainable_variables

Slayers
Tnon_trainable_variables
 
 
 
≠
Ulayer_regularization_losses
regularization_losses
Vmetrics
	variables
Wlayer_metrics
trainable_variables

Xlayers
Ynon_trainable_variables
О
Z
state_size

@kernel
Arecurrent_kernel
Bbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
 
 

@0
A1
B2

@0
A1
B2
є
_layer_regularization_losses
!regularization_losses
"trainable_variables
`metrics
alayer_metrics

bstates

clayers
#	variables
dnon_trainable_variables
О
e
state_size

Ckernel
Drecurrent_kernel
Ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
 
 

C0
D1
E2

C0
D1
E2
є
jlayer_regularization_losses
'regularization_losses
(trainable_variables
kmetrics
llayer_metrics

mstates

nlayers
)	variables
onon_trainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
≠
player_regularization_losses
-regularization_losses
qmetrics
.	variables
rlayer_metrics
/trainable_variables

slayers
tnon_trainable_variables
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
≠
ulayer_regularization_losses
3regularization_losses
vmetrics
4	variables
wlayer_metrics
5trainable_variables

xlayers
ynon_trainable_variables
 
 
 
≠
zlayer_regularization_losses
7regularization_losses
{metrics
8	variables
|layer_metrics
9trainable_variables

}layers
~non_trainable_variables
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
VARIABLE_VALUElstm_3/lstm_cell_3/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_3/lstm_cell_3/recurrent_kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_3/lstm_cell_3/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_4/lstm_cell_4/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_4/lstm_cell_4/recurrent_kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_4/lstm_cell_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
 

0
 
8
0
1
2
3
4
5
6
7
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

@0
A1
B2

@0
A1
B2
≤
 Аlayer_regularization_losses
[regularization_losses
Бmetrics
\	variables
Вlayer_metrics
]trainable_variables
Гlayers
Дnon_trainable_variables
 
 
 
 

0
 
 
 

C0
D1
E2

C0
D1
E2
≤
 Еlayer_regularization_losses
fregularization_losses
Жmetrics
g	variables
Зlayer_metrics
htrainable_variables
Иlayers
Йnon_trainable_variables
 
 
 
 

%0
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

Кtotal

Лcount
М	variables
Н	keras_api
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
К0
Л1

М	variables
|z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_3/lstm_cell_3/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_3/lstm_cell_3/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
З
serving_default_conv1d_inputPlaceholder*+
_output_shapes
:€€€€€€€€€*
dtype0* 
shape:€€€€€€€€€
Б
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv1d_inputconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biaslstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biaslstm_4/lstm_cell_4/kernellstm_4/lstm_cell_4/bias#lstm_4/lstm_cell_4/recurrent_kerneldense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_154397
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
µ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_3/lstm_cell_3/kernel/Read/ReadVariableOp7lstm_3/lstm_cell_3/recurrent_kernel/Read/ReadVariableOp+lstm_3/lstm_cell_3/bias/Read/ReadVariableOp-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_4/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/m/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp4Adam/lstm_3/lstm_cell_3/kernel/v/Read/ReadVariableOp>Adam/lstm_3/lstm_cell_3/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_3/lstm_cell_3/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpConst*>
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
__inference__traced_save_157886
№
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_3/lstm_cell_3/kernel#lstm_3/lstm_cell_3/recurrent_kernellstm_3/lstm_cell_3/biaslstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biastotalcountAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/m Adam/lstm_3/lstm_cell_3/kernel/m*Adam/lstm_3/lstm_cell_3/recurrent_kernel/mAdam/lstm_3/lstm_cell_3/bias/m Adam/lstm_4/lstm_cell_4/kernel/m*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mAdam/lstm_4/lstm_cell_4/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v Adam/lstm_3/lstm_cell_3/kernel/v*Adam/lstm_3/lstm_cell_3/recurrent_kernel/vAdam/lstm_3/lstm_cell_3/bias/v Adam/lstm_4/lstm_cell_4/kernel/v*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vAdam/lstm_4/lstm_cell_4/bias/v*=
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
"__inference__traced_restore_158043іЭ7
п
Х
(__inference_dense_5_layer_call_fn_157334

inputs
unknown:@
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
C__inference_dense_5_layer_call_and_return_conditional_losses_1534712
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
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
•

ф
C__inference_dense_5_layer_call_and_return_conditional_losses_153471

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
®

ѕ
lstm_3_while_cond_154557*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1B
>lstm_3_while_lstm_3_while_cond_154557___redundant_placeholder0B
>lstm_3_while_lstm_3_while_cond_154557___redundant_placeholder1B
>lstm_3_while_lstm_3_while_cond_154557___redundant_placeholder2B
>lstm_3_while_lstm_3_while_cond_154557___redundant_placeholder3
lstm_3_while_identity
У
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
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
£
e
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_155507

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
:€€€€€€€€€
@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
@:S O
+
_output_shapes
:€€€€€€€€€
@
 
_user_specified_nameinputs
ѓ
”
%sequential_1_lstm_4_while_cond_151378D
@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counterJ
Fsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations)
%sequential_1_lstm_4_while_placeholder+
'sequential_1_lstm_4_while_placeholder_1+
'sequential_1_lstm_4_while_placeholder_2+
'sequential_1_lstm_4_while_placeholder_3F
Bsequential_1_lstm_4_while_less_sequential_1_lstm_4_strided_slice_1\
Xsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_151378___redundant_placeholder0\
Xsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_151378___redundant_placeholder1\
Xsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_151378___redundant_placeholder2\
Xsequential_1_lstm_4_while_sequential_1_lstm_4_while_cond_151378___redundant_placeholder3&
"sequential_1_lstm_4_while_identity
‘
sequential_1/lstm_4/while/LessLess%sequential_1_lstm_4_while_placeholderBsequential_1_lstm_4_while_less_sequential_1_lstm_4_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_4/while/LessЩ
"sequential_1/lstm_4/while/IdentityIdentity"sequential_1/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_4/while/Identity"Q
"sequential_1_lstm_4_while_identity+sequential_1/lstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
њ%
№
while_body_151645
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_3_151669_0:	@А-
while_lstm_cell_3_151671_0:	 А)
while_lstm_cell_3_151673_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_3_151669:	@А+
while_lstm_cell_3_151671:	 А'
while_lstm_cell_3_151673:	АИҐ)while/lstm_cell_3/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemџ
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_151669_0while_lstm_cell_3_151671_0while_lstm_cell_3_151673_0*
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1516312+
)while/lstm_cell_3/StatefulPartitionedCallц
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
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
while_lstm_cell_3_151669while_lstm_cell_3_151669_0"6
while_lstm_cell_3_151671while_lstm_cell_3_151671_0"6
while_lstm_cell_3_151673while_lstm_cell_3_151673_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
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
®

ѕ
lstm_4_while_cond_155225*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_155225___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_155225___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_155225___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_155225___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
’
√
while_cond_153716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_153716___redundant_placeholder04
0while_while_cond_153716___redundant_placeholder14
0while_while_cond_153716___redundant_placeholder24
0while_while_cond_153716___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
И
a
E__inference_reshape_2_layer_call_and_return_conditional_losses_153490

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
Е
Ъ
)__inference_conv1d_1_layer_call_fn_155465

inputs
unknown: @
	unknown_0:@
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1530212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€
@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
„
ґ
'__inference_lstm_3_layer_call_fn_155529
inputs_0
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1519242
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0
“[
Ц
B__inference_lstm_3_layer_call_and_return_conditional_losses_155853
inputs_0=
*lstm_cell_3_matmul_readvariableop_resource:	@А?
,lstm_cell_3_matmul_1_readvariableop_resource:	 А:
+lstm_cell_3_biasadd_readvariableop_resource:	А
identityИҐ"lstm_cell_3/BiasAdd/ReadVariableOpҐ!lstm_cell_3/MatMul/ReadVariableOpҐ#lstm_cell_3/MatMul_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp™
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMulЄ
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¶
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMul_1Ь
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/BiasAdd|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimп
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_cell_3/splitГ
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/SigmoidЗ
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_1И
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/ReluШ
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_1Н
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/add_1З
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Relu_1Ь
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_2П
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
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
while_body_155769*
condR
while_cond_155768*K
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
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity≈
NoOpNoOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€@: : : 2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0
¬>
«
while_body_155769
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_3_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_3_matmul_readvariableop_resource:	@АE
2while_lstm_cell_3_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ(while/lstm_cell_3/BiasAdd/ReadVariableOpҐ'while/lstm_cell_3/MatMul/ReadVariableOpҐ)while/lstm_cell_3/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem∆
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp‘
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMulћ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOpљ
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMul_1і
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/add≈
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpЅ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/BiasAddИ
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dimЗ
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
while/lstm_cell_3/splitХ
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/SigmoidЩ
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_1Э
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mulМ
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu∞
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_1•
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/add_1Щ
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_2Л
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu_1і
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_2я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5џ

while/NoOpNoOp)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*"
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
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
C__inference_dense_5_layer_call_and_return_conditional_losses_157344

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
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
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
’
√
while_cond_152323
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_152323___redundant_placeholder04
0while_while_cond_152323___redundant_placeholder14
0while_while_cond_152323___redundant_placeholder24
0while_while_cond_152323___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
У[
Ф
B__inference_lstm_3_layer_call_and_return_conditional_losses_156155

inputs=
*lstm_cell_3_matmul_readvariableop_resource:	@А?
,lstm_cell_3_matmul_1_readvariableop_resource:	 А:
+lstm_cell_3_biasadd_readvariableop_resource:	А
identityИҐ"lstm_cell_3/BiasAdd/ReadVariableOpҐ!lstm_cell_3/MatMul/ReadVariableOpҐ#lstm_cell_3/MatMul_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp™
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMulЄ
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¶
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMul_1Ь
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/BiasAdd|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimп
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_cell_3/splitГ
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/SigmoidЗ
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_1И
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/ReluШ
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_1Н
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/add_1З
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Relu_1Ь
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_2П
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
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
while_body_156071*
condR
while_cond_156070*K
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
:€€€€€€€€€ *
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
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

Identity≈
NoOpNoOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€@: : : 2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Тv
и
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_157705

inputs
states_0
states_10
split_readvariableop_resource:	 А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2эхе2&
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
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shapeў
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ћГ√2(
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
:€€€€€€€€€@2
dropout_1/GreaterEqualЕ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_1/CastВ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/ShapeЎ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2≠щZ2(
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
:€€€€€€€€€@2
dropout_2/GreaterEqualЕ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_2/CastВ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/ShapeЎ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ѕа&2(
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
:€€€€€€€€€@2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_3/Mul_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_3`
mulMulstates_0dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mulf
mul_1Mulstates_0dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1f
mul_2Mulstates_0dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2
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

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2
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

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

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
?:€€€€€€€€€ :€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
¬>
«
while_body_155618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_3_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_3_matmul_readvariableop_resource:	@АE
2while_lstm_cell_3_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ(while/lstm_cell_3/BiasAdd/ReadVariableOpҐ'while/lstm_cell_3/MatMul/ReadVariableOpҐ)while/lstm_cell_3/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem∆
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp‘
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMulћ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOpљ
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMul_1і
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/add≈
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpЅ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/BiasAddИ
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dimЗ
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
while/lstm_cell_3/splitХ
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/SigmoidЩ
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_1Э
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mulМ
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu∞
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_1•
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/add_1Щ
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_2Л
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu_1і
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_2я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5џ

while/NoOpNoOp)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*"
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
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
“[
Ц
B__inference_lstm_3_layer_call_and_return_conditional_losses_155702
inputs_0=
*lstm_cell_3_matmul_readvariableop_resource:	@А?
,lstm_cell_3_matmul_1_readvariableop_resource:	 А:
+lstm_cell_3_biasadd_readvariableop_resource:	А
identityИҐ"lstm_cell_3/BiasAdd/ReadVariableOpҐ!lstm_cell_3/MatMul/ReadVariableOpҐ#lstm_cell_3/MatMul_1/ReadVariableOpҐwhileF
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
 :€€€€€€€€€€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp™
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMulЄ
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¶
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMul_1Ь
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/BiasAdd|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimп
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_cell_3/splitГ
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/SigmoidЗ
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_1И
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/ReluШ
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_1Н
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/add_1З
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Relu_1Ь
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_2П
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
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
while_body_155618*
condR
while_cond_155617*K
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
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity≈
NoOpNoOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€@: : : 2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0
і
х
,__inference_lstm_cell_4_layer_call_fn_157494

inputs
states_0
states_1
unknown:	 А
	unknown_0:	А
	unknown_1:	@А
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
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1523102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

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
?:€€€€€€€€€ :€€€€€€€€€@:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
’
√
while_cond_152620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_152620___redundant_placeholder04
0while_while_cond_152620___redundant_placeholder14
0while_while_cond_152620___redundant_placeholder24
0while_while_cond_152620___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
і
х
,__inference_lstm_cell_3_layer_call_fn_157407

inputs
states_0
states_1
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1517772
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
?:€€€€€€€€€@:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
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
ƒ~
Ш	
while_body_156865
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	 АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	 А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	@АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2
while/lstm_cell_4/ones_likeИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_3Ґ
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul¶
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_1¶
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_2¶
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   2)
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

:@@*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2)
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2D
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
Љ
ґ
'__inference_lstm_4_layer_call_fn_156172
inputs_0
unknown:	 А
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1523992
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
сћ
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_153882

inputs<
)lstm_cell_4_split_readvariableop_resource:	 А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
:€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout/MulД
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeц
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2нчE22
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
:€€€€€€€€€@2"
 lstm_cell_4/dropout/GreaterEqual£
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout/Cast™
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_1/MulИ
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shapeь
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2б≠]24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_1/GreaterEqual©
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_1/Cast≤
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_2/MulИ
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeэ
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2с≈К24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_2/GreaterEqual©
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_2/Cast≤
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_3/MulИ
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeэ
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2З∆∆24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_3/GreaterEqual©
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_3/Cast≤
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_3К
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulР
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Р
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Р
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_153717*
condR
while_cond_153716*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@*
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
:€€€€€€€€€@2
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
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2z
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
:€€€€€€€€€ 
 
_user_specified_nameinputs
ї∞
Ш	
while_body_156590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	 АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	 А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	@АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/lstm_cell_4/dropout/MulЦ
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/ShapeЙ
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ЫВћ28
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
:€€€€€€€€€@2(
&while/lstm_cell_4/dropout/GreaterEqualµ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2 
while/lstm_cell_4/dropout/Cast¬
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_1/MulЪ
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/ShapeП
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ЫЙѕ2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_1/GreaterEqualї
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_1/Cast 
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_2/MulЪ
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/ShapeП
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ыЛй2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_2/GreaterEqualї
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_2/Cast 
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_3/MulЪ
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/ShapeО
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2џЫJ2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_3/GreaterEqualї
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_3/Cast 
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!while/lstm_cell_4/dropout_3/Mul_1И
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_3°
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mulІ
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_1І
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_2І
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   2)
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

:@@*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2)
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2D
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
ђ
У
D__inference_conv1d_1_layer_call_and_return_conditional_losses_153021

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@*
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€
@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€
@2

IdentityМ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ѓ
”
%sequential_1_lstm_3_while_cond_151188D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3F
Bsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1\
Xsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_151188___redundant_placeholder0\
Xsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_151188___redundant_placeholder1\
Xsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_151188___redundant_placeholder2\
Xsequential_1_lstm_3_while_sequential_1_lstm_3_while_cond_151188___redundant_placeholder3&
"sequential_1_lstm_3_while_identity
‘
sequential_1/lstm_3/while/LessLess%sequential_1_lstm_3_while_placeholderBsequential_1_lstm_3_while_less_sequential_1_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/while/LessЩ
"sequential_1/lstm_3/while/IdentityIdentity"sequential_1/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_1/lstm_3/while/Identity"Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0*(
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
¬>
«
while_body_153971
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_3_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_3_matmul_readvariableop_resource:	@АE
2while_lstm_cell_3_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ(while/lstm_cell_3/BiasAdd/ReadVariableOpҐ'while/lstm_cell_3/MatMul/ReadVariableOpҐ)while/lstm_cell_3/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem∆
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp‘
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMulћ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOpљ
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMul_1і
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/add≈
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpЅ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/BiasAddИ
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dimЗ
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
while/lstm_cell_3/splitХ
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/SigmoidЩ
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_1Э
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mulМ
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu∞
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_1•
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/add_1Щ
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_2Л
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu_1і
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_2я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5џ

while/NoOpNoOp)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*"
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
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
Ц
√
B__inference_conv1d_layer_call_and_return_conditional_losses_155456

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€ 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
Relu“
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mulq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ 2

IdentityЊ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv1d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_153101
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_153101___redundant_placeholder04
0while_while_cond_153101___redundant_placeholder14
0while_while_cond_153101___redundant_placeholder24
0while_while_cond_153101___redundant_placeholder3
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
„Я
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_153436

inputs<
)lstm_cell_4_split_readvariableop_resource:	 А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
:€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
lstm_cell_4/ones_like|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_3Л
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulП
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1П
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_153303*
condR
while_cond_153302*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@*
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
:€€€€€€€€€@2
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
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2z
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
:€€€€€€€€€ 
 
_user_specified_nameinputs
п
Х
(__inference_dense_4_layer_call_fn_157314

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1534552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ђ
У
D__inference_conv1d_1_layer_call_and_return_conditional_losses_155481

inputsA
+conv1d_expanddims_1_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv1d/ExpandDimsЄ
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
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@*
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@*
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€
@2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
Reluq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€
@2

IdentityМ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
ЅH
І

lstm_3_while_body_155004*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0:	@АN
;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АI
:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource:	@АL
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource:	 АG
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpҐ.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpҐ0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp—
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeэ
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItemџ
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype020
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpр
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
lstm_3/while/lstm_cell_3/MatMulб
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpў
!lstm_3/while/lstm_cell_3/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!lstm_3/while/lstm_cell_3/MatMul_1–
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/MatMul:product:0+lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/while/lstm_cell_3/addЏ
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpЁ
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd lstm_3/while/lstm_cell_3/add:z:07lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 lstm_3/while/lstm_cell_3/BiasAddЦ
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim£
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:0)lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2 
lstm_3/while/lstm_cell_3/split™
 lstm_3/while/lstm_cell_3/SigmoidSigmoid'lstm_3/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_3/while/lstm_cell_3/SigmoidЃ
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_3/while/lstm_cell_3/Sigmoid_1є
lstm_3/while/lstm_cell_3/mulMul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/lstm_cell_3/mul°
lstm_3/while/lstm_cell_3/ReluRelu'lstm_3/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/lstm_cell_3/Reluћ
lstm_3/while/lstm_cell_3/mul_1Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0+lstm_3/while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_3/while/lstm_cell_3/mul_1Ѕ
lstm_3/while/lstm_cell_3/add_1AddV2 lstm_3/while/lstm_cell_3/mul:z:0"lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_3/while/lstm_cell_3/add_1Ѓ
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_3/while/lstm_cell_3/Sigmoid_2†
lstm_3/while/lstm_cell_3/Relu_1Relu"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_3/while/lstm_cell_3/Relu_1–
lstm_3/while/lstm_cell_3/mul_2Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0-lstm_3/while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_3/while/lstm_cell_3/mul_2В
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/yЕ
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/yЩ
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1З
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity°
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1Й
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2ґ
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3®
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_2:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/Identity_4®
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_1:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/Identity_5ю
lstm_3/while/NoOpNoOp0^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_3/while/NoOp"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"ƒ
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2b
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
while_cond_151644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_151644___redundant_placeholder04
0while_while_cond_151644___redundant_placeholder14
0while_while_cond_151644___redundant_placeholder24
0while_while_cond_151644___redundant_placeholder3
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
й=
н
H__inference_sequential_1_layer_call_and_return_conditional_losses_153505

inputs#
conv1d_153000: 
conv1d_153002: %
conv1d_1_153022: @
conv1d_1_153024:@ 
lstm_3_153187:	@А 
lstm_3_153189:	 А
lstm_3_153191:	А 
lstm_4_153437:	 А
lstm_4_153439:	А 
lstm_4_153441:	@А 
dense_4_153456:@@
dense_4_153458:@ 
dense_5_153472:@
dense_5_153474:
identityИҐconv1d/StatefulPartitionedCallҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpҐ conv1d_1/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpО
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_153000conv1d_153002*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1529992 
conv1d/StatefulPartitionedCallє
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_153022conv1d_1_153024*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1530212"
 conv1d_1/StatefulPartitionedCallК
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_1530342
max_pooling1d/PartitionedCallњ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0lstm_3_153187lstm_3_153189lstm_3_153191*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1531862 
lstm_3/StatefulPartitionedCallЉ
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_153437lstm_4_153439lstm_4_153441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1534362 
lstm_4/StatefulPartitionedCall∞
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_153456dense_4_153458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1534552!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_153472dense_5_153474*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_1534712!
dense_5/StatefulPartitionedCallэ
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
E__inference_reshape_2_layer_call_and_return_conditional_losses_1534902
reshape_2/PartitionedCallі
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_153000*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_153437*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_2/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityИ
NoOpNoOp^conv1d/StatefulPartitionedCall0^conv1d/kernel/Regularizer/Square/ReadVariableOp!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Б
ф
C__inference_dense_4_layer_call_and_return_conditional_losses_157325

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ч≤
‘
%sequential_1_lstm_4_while_body_151379D
@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counterJ
Fsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations)
%sequential_1_lstm_4_while_placeholder+
'sequential_1_lstm_4_while_placeholder_1+
'sequential_1_lstm_4_while_placeholder_2+
'sequential_1_lstm_4_while_placeholder_3C
?sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1_0
{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0X
Esequential_1_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	 АV
Gsequential_1_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	АR
?sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource_0:	@А&
"sequential_1_lstm_4_while_identity(
$sequential_1_lstm_4_while_identity_1(
$sequential_1_lstm_4_while_identity_2(
$sequential_1_lstm_4_while_identity_3(
$sequential_1_lstm_4_while_identity_4(
$sequential_1_lstm_4_while_identity_5A
=sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1}
ysequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensorV
Csequential_1_lstm_4_while_lstm_cell_4_split_readvariableop_resource:	 АT
Esequential_1_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	АP
=sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource:	@АИҐ4sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOpҐ6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_1Ґ6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_2Ґ6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_3Ґ:sequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOpҐ<sequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpл
Ksequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2M
Ksequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeЋ
=sequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_4_while_placeholderTsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype02?
=sequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem≈
5sequential_1/lstm_4/while/lstm_cell_4/ones_like/ShapeShape'sequential_1_lstm_4_while_placeholder_2*
T0*
_output_shapes
:27
5sequential_1/lstm_4/while/lstm_cell_4/ones_like/Shape≥
5sequential_1/lstm_4/while/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?27
5sequential_1/lstm_4/while/lstm_cell_4/ones_like/ConstЬ
/sequential_1/lstm_4/while/lstm_cell_4/ones_likeFill>sequential_1/lstm_4/while/lstm_cell_4/ones_like/Shape:output:0>sequential_1/lstm_4/while/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/sequential_1/lstm_4/while/lstm_cell_4/ones_like∞
5sequential_1/lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_4/while/lstm_cell_4/split/split_dim€
:sequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOpEsequential_1_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02<
:sequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOpњ
+sequential_1/lstm_4/while/lstm_cell_4/splitSplit>sequential_1/lstm_4/while/lstm_cell_4/split/split_dim:output:0Bsequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2-
+sequential_1/lstm_4/while/lstm_cell_4/splitФ
,sequential_1/lstm_4/while/lstm_cell_4/MatMulMatMulDsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2.
,sequential_1/lstm_4/while/lstm_cell_4/MatMulШ
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_1MatMulDsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_1Ш
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_2MatMulDsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_2Ш
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_3MatMulDsequential_1/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:04sequential_1/lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_3і
7sequential_1/lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_1/lstm_4/while/lstm_cell_4/split_1/split_dimБ
<sequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOpGsequential_1_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpЈ
-sequential_1/lstm_4/while/lstm_cell_4/split_1Split@sequential_1/lstm_4/while/lstm_cell_4/split_1/split_dim:output:0Dsequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2/
-sequential_1/lstm_4/while/lstm_cell_4/split_1Л
-sequential_1/lstm_4/while/lstm_cell_4/BiasAddBiasAdd6sequential_1/lstm_4/while/lstm_cell_4/MatMul:product:06sequential_1/lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2/
-sequential_1/lstm_4/while/lstm_cell_4/BiasAddС
/sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd8sequential_1/lstm_4/while/lstm_cell_4/MatMul_1:product:06sequential_1/lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@21
/sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_1С
/sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd8sequential_1/lstm_4/while/lstm_cell_4/MatMul_2:product:06sequential_1/lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@21
/sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_2С
/sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd8sequential_1/lstm_4/while/lstm_cell_4/MatMul_3:product:06sequential_1/lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@21
/sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_3т
)sequential_1/lstm_4/while/lstm_cell_4/mulMul'sequential_1_lstm_4_while_placeholder_28sequential_1/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/while/lstm_cell_4/mulц
+sequential_1/lstm_4/while/lstm_cell_4/mul_1Mul'sequential_1_lstm_4_while_placeholder_28sequential_1/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/mul_1ц
+sequential_1/lstm_4/while/lstm_cell_4/mul_2Mul'sequential_1_lstm_4_while_placeholder_28sequential_1/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/mul_2ц
+sequential_1/lstm_4/while/lstm_cell_4/mul_3Mul'sequential_1_lstm_4_while_placeholder_28sequential_1/lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/mul_3н
4sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp?sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype026
4sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp«
9sequential_1/lstm_4/while/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2;
9sequential_1/lstm_4/while/lstm_cell_4/strided_slice/stackЋ
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   2=
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack_1Ћ
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2=
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack_2а
3sequential_1/lstm_4/while/lstm_cell_4/strided_sliceStridedSlice<sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp:value:0Bsequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack:output:0Dsequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack_1:output:0Dsequential_1/lstm_4/while/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask25
3sequential_1/lstm_4/while/lstm_cell_4/strided_sliceЙ
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_4MatMul-sequential_1/lstm_4/while/lstm_cell_4/mul:z:0<sequential_1/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_4Г
)sequential_1/lstm_4/while/lstm_cell_4/addAddV26sequential_1/lstm_4/while/lstm_cell_4/BiasAdd:output:08sequential_1/lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/while/lstm_cell_4/add 
-sequential_1/lstm_4/while/lstm_cell_4/SigmoidSigmoid-sequential_1/lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2/
-sequential_1/lstm_4/while/lstm_cell_4/Sigmoidс
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp?sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype028
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_1Ћ
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2=
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stackѕ
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2?
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1ѕ
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2м
5sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1StridedSlice>sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_1:value:0Dsequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack:output:0Fsequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack_1:output:0Fsequential_1/lstm_4/while/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask27
5sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1Н
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_5MatMul/sequential_1/lstm_4/while/lstm_cell_4/mul_1:z:0>sequential_1/lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_5Й
+sequential_1/lstm_4/while/lstm_cell_4/add_1AddV28sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_1:output:08sequential_1/lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/add_1–
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid/sequential_1/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_1с
+sequential_1/lstm_4/while/lstm_cell_4/mul_4Mul3sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_1:y:0'sequential_1_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/mul_4с
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp?sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype028
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_2Ћ
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2=
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stackѕ
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2?
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1ѕ
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2м
5sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2StridedSlice>sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_2:value:0Dsequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack:output:0Fsequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack_1:output:0Fsequential_1/lstm_4/while/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask27
5sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2Н
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_6MatMul/sequential_1/lstm_4/while/lstm_cell_4/mul_2:z:0>sequential_1/lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_6Й
+sequential_1/lstm_4/while/lstm_cell_4/add_2AddV28sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_2:output:08sequential_1/lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/add_2√
*sequential_1/lstm_4/while/lstm_cell_4/ReluRelu/sequential_1/lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2,
*sequential_1/lstm_4/while/lstm_cell_4/ReluА
+sequential_1/lstm_4/while/lstm_cell_4/mul_5Mul1sequential_1/lstm_4/while/lstm_cell_4/Sigmoid:y:08sequential_1/lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/mul_5ч
+sequential_1/lstm_4/while/lstm_cell_4/add_3AddV2/sequential_1/lstm_4/while/lstm_cell_4/mul_4:z:0/sequential_1/lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/add_3с
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp?sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype028
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_3Ћ
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2=
;sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stackѕ
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2?
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1ѕ
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2?
=sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2м
5sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3StridedSlice>sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_3:value:0Dsequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack:output:0Fsequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack_1:output:0Fsequential_1/lstm_4/while/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask27
5sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3Н
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_7MatMul/sequential_1/lstm_4/while/lstm_cell_4/mul_3:z:0>sequential_1/lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.sequential_1/lstm_4/while/lstm_cell_4/MatMul_7Й
+sequential_1/lstm_4/while/lstm_cell_4/add_4AddV28sequential_1/lstm_4/while/lstm_cell_4/BiasAdd_3:output:08sequential_1/lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/add_4–
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid/sequential_1/lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@21
/sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_2«
,sequential_1/lstm_4/while/lstm_cell_4/Relu_1Relu/sequential_1/lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2.
,sequential_1/lstm_4/while/lstm_cell_4/Relu_1Д
+sequential_1/lstm_4/while/lstm_cell_4/mul_6Mul3sequential_1/lstm_4/while/lstm_cell_4/Sigmoid_2:y:0:sequential_1/lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2-
+sequential_1/lstm_4/while/lstm_cell_4/mul_6√
>sequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_4_while_placeholder_1%sequential_1_lstm_4_while_placeholder/sequential_1/lstm_4/while/lstm_cell_4/mul_6:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItemД
sequential_1/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_4/while/add/yє
sequential_1/lstm_4/while/addAddV2%sequential_1_lstm_4_while_placeholder(sequential_1/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_4/while/addИ
!sequential_1/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_4/while/add_1/yЏ
sequential_1/lstm_4/while/add_1AddV2@sequential_1_lstm_4_while_sequential_1_lstm_4_while_loop_counter*sequential_1/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_4/while/add_1ї
"sequential_1/lstm_4/while/IdentityIdentity#sequential_1/lstm_4/while/add_1:z:0^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_1/lstm_4/while/Identityв
$sequential_1/lstm_4/while/Identity_1IdentityFsequential_1_lstm_4_while_sequential_1_lstm_4_while_maximum_iterations^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_4/while/Identity_1љ
$sequential_1/lstm_4/while/Identity_2Identity!sequential_1/lstm_4/while/add:z:0^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_4/while/Identity_2к
$sequential_1/lstm_4/while/Identity_3IdentityNsequential_1/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_4/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_4/while/Identity_3№
$sequential_1/lstm_4/while/Identity_4Identity/sequential_1/lstm_4/while/lstm_cell_4/mul_6:z:0^sequential_1/lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2&
$sequential_1/lstm_4/while/Identity_4№
$sequential_1/lstm_4/while/Identity_5Identity/sequential_1/lstm_4/while/lstm_cell_4/add_3:z:0^sequential_1/lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2&
$sequential_1/lstm_4/while/Identity_5а
sequential_1/lstm_4/while/NoOpNoOp5^sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp7^sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_17^sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_27^sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_3;^sequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOp=^sequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_1/lstm_4/while/NoOp"Q
"sequential_1_lstm_4_while_identity+sequential_1/lstm_4/while/Identity:output:0"U
$sequential_1_lstm_4_while_identity_1-sequential_1/lstm_4/while/Identity_1:output:0"U
$sequential_1_lstm_4_while_identity_2-sequential_1/lstm_4/while/Identity_2:output:0"U
$sequential_1_lstm_4_while_identity_3-sequential_1/lstm_4/while/Identity_3:output:0"U
$sequential_1_lstm_4_while_identity_4-sequential_1/lstm_4/while/Identity_4:output:0"U
$sequential_1_lstm_4_while_identity_5-sequential_1/lstm_4/while/Identity_5:output:0"А
=sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource?sequential_1_lstm_4_while_lstm_cell_4_readvariableop_resource_0"Р
Esequential_1_lstm_4_while_lstm_cell_4_split_1_readvariableop_resourceGsequential_1_lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0"М
Csequential_1_lstm_4_while_lstm_cell_4_split_readvariableop_resourceEsequential_1_lstm_4_while_lstm_cell_4_split_readvariableop_resource_0"А
=sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1?sequential_1_lstm_4_while_sequential_1_lstm_4_strided_slice_1_0"ш
ysequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2l
4sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp4sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp2p
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_16sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_12p
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_26sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_22p
6sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_36sequential_1/lstm_4/while/lstm_cell_4/ReadVariableOp_32x
:sequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOp:sequential_1/lstm_4/while/lstm_cell_4/split/ReadVariableOp2|
<sequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp<sequential_1/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
¬>
«
while_body_155920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_3_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_3_matmul_readvariableop_resource:	@АE
2while_lstm_cell_3_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ(while/lstm_cell_3/BiasAdd/ReadVariableOpҐ'while/lstm_cell_3/MatMul/ReadVariableOpҐ)while/lstm_cell_3/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem∆
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp‘
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMulћ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOpљ
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMul_1і
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/add≈
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpЅ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/BiasAddИ
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dimЗ
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
while/lstm_cell_3/splitХ
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/SigmoidЩ
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_1Э
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mulМ
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu∞
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_1•
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/add_1Щ
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_2Л
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu_1і
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_2я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5џ

while/NoOpNoOp)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*"
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
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
Е[
«
%sequential_1_lstm_3_while_body_151189D
@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counterJ
Fsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations)
%sequential_1_lstm_3_while_placeholder+
'sequential_1_lstm_3_while_placeholder_1+
'sequential_1_lstm_3_while_placeholder_2+
'sequential_1_lstm_3_while_placeholder_3C
?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0
{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0:	@А[
Hsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АV
Gsequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0:	А&
"sequential_1_lstm_3_while_identity(
$sequential_1_lstm_3_while_identity_1(
$sequential_1_lstm_3_while_identity_2(
$sequential_1_lstm_3_while_identity_3(
$sequential_1_lstm_3_while_identity_4(
$sequential_1_lstm_3_while_identity_5A
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1}
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensorW
Dsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource:	@АY
Fsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource:	 АT
Esequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpҐ;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpҐ=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpл
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2M
Ksequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeЋ
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0%sequential_1_lstm_3_while_placeholderTsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02?
=sequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItemВ
;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOpFsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02=
;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp§
,sequential_1/lstm_3/while/lstm_cell_3/MatMulMatMulDsequential_1/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2.
,sequential_1/lstm_3/while/lstm_cell_3/MatMulИ
=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOpHsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02?
=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpН
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_1MatMul'sequential_1_lstm_3_while_placeholder_2Esequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А20
.sequential_1/lstm_3/while/lstm_cell_3/MatMul_1Д
)sequential_1/lstm_3/while/lstm_cell_3/addAddV26sequential_1/lstm_3/while/lstm_cell_3/MatMul:product:08sequential_1/lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2+
)sequential_1/lstm_3/while/lstm_cell_3/addБ
<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOpGsequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02>
<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpС
-sequential_1/lstm_3/while/lstm_cell_3/BiasAddBiasAdd-sequential_1/lstm_3/while/lstm_cell_3/add:z:0Dsequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2/
-sequential_1/lstm_3/while/lstm_cell_3/BiasAdd∞
5sequential_1/lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_1/lstm_3/while/lstm_cell_3/split/split_dim„
+sequential_1/lstm_3/while/lstm_cell_3/splitSplit>sequential_1/lstm_3/while/lstm_cell_3/split/split_dim:output:06sequential_1/lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2-
+sequential_1/lstm_3/while/lstm_cell_3/split—
-sequential_1/lstm_3/while/lstm_cell_3/SigmoidSigmoid4sequential_1/lstm_3/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2/
-sequential_1/lstm_3/while/lstm_cell_3/Sigmoid’
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid4sequential_1/lstm_3/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_1н
)sequential_1/lstm_3/while/lstm_cell_3/mulMul3sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_1:y:0'sequential_1_lstm_3_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_1/lstm_3/while/lstm_cell_3/mul»
*sequential_1/lstm_3/while/lstm_cell_3/ReluRelu4sequential_1/lstm_3/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2,
*sequential_1/lstm_3/while/lstm_cell_3/ReluА
+sequential_1/lstm_3/while/lstm_cell_3/mul_1Mul1sequential_1/lstm_3/while/lstm_cell_3/Sigmoid:y:08sequential_1/lstm_3/while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_1/lstm_3/while/lstm_cell_3/mul_1х
+sequential_1/lstm_3/while/lstm_cell_3/add_1AddV2-sequential_1/lstm_3/while/lstm_cell_3/mul:z:0/sequential_1/lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_1/lstm_3/while/lstm_cell_3/add_1’
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid4sequential_1/lstm_3/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 21
/sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_2«
,sequential_1/lstm_3/while/lstm_cell_3/Relu_1Relu/sequential_1/lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2.
,sequential_1/lstm_3/while/lstm_cell_3/Relu_1Д
+sequential_1/lstm_3/while/lstm_cell_3/mul_2Mul3sequential_1/lstm_3/while/lstm_cell_3/Sigmoid_2:y:0:sequential_1/lstm_3/while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2-
+sequential_1/lstm_3/while/lstm_cell_3/mul_2√
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_1_lstm_3_while_placeholder_1%sequential_1_lstm_3_while_placeholder/sequential_1/lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItemД
sequential_1/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_1/lstm_3/while/add/yє
sequential_1/lstm_3/while/addAddV2%sequential_1_lstm_3_while_placeholder(sequential_1/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/while/addИ
!sequential_1/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_1/lstm_3/while/add_1/yЏ
sequential_1/lstm_3/while/add_1AddV2@sequential_1_lstm_3_while_sequential_1_lstm_3_while_loop_counter*sequential_1/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/while/add_1ї
"sequential_1/lstm_3/while/IdentityIdentity#sequential_1/lstm_3/while/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_1/lstm_3/while/Identityв
$sequential_1/lstm_3/while/Identity_1IdentityFsequential_1_lstm_3_while_sequential_1_lstm_3_while_maximum_iterations^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_1љ
$sequential_1/lstm_3/while/Identity_2Identity!sequential_1/lstm_3/while/add:z:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_2к
$sequential_1/lstm_3/while/Identity_3IdentityNsequential_1/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_1/lstm_3/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_1/lstm_3/while/Identity_3№
$sequential_1/lstm_3/while/Identity_4Identity/sequential_1/lstm_3/while/lstm_cell_3/mul_2:z:0^sequential_1/lstm_3/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential_1/lstm_3/while/Identity_4№
$sequential_1/lstm_3/while/Identity_5Identity/sequential_1/lstm_3/while/lstm_cell_3/add_1:z:0^sequential_1/lstm_3/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential_1/lstm_3/while/Identity_5њ
sequential_1/lstm_3/while/NoOpNoOp=^sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp<^sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp>^sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_1/lstm_3/while/NoOp"Q
"sequential_1_lstm_3_while_identity+sequential_1/lstm_3/while/Identity:output:0"U
$sequential_1_lstm_3_while_identity_1-sequential_1/lstm_3/while/Identity_1:output:0"U
$sequential_1_lstm_3_while_identity_2-sequential_1/lstm_3/while/Identity_2:output:0"U
$sequential_1_lstm_3_while_identity_3-sequential_1/lstm_3/while/Identity_3:output:0"U
$sequential_1_lstm_3_while_identity_4-sequential_1/lstm_3/while/Identity_4:output:0"U
$sequential_1_lstm_3_while_identity_5-sequential_1/lstm_3/while/Identity_5:output:0"Р
Esequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resourceGsequential_1_lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"Т
Fsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resourceHsequential_1_lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"О
Dsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resourceFsequential_1_lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"А
=sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1?sequential_1_lstm_3_while_sequential_1_lstm_3_strided_slice_1_0"ш
ysequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor{sequential_1_lstm_3_while_tensorarrayv2read_tensorlistgetitem_sequential_1_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2|
<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp<sequential_1/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2z
;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp;sequential_1/lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2~
=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp=sequential_1/lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
Ї∞
Ш	
while_body_157140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	 АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	 А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	@АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/lstm_cell_4/dropout/MulЦ
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/ShapeИ
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2яЬE28
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
:€€€€€€€€€@2(
&while/lstm_cell_4/dropout/GreaterEqualµ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2 
while/lstm_cell_4/dropout/Cast¬
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_1/MulЪ
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/ShapeП
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ћо»2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_1/GreaterEqualї
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_1/Cast 
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_2/MulЪ
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/ShapeО
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2зІG2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_2/GreaterEqualї
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_2/Cast 
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_3/MulЪ
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/ShapeП
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Рњ™2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_3/GreaterEqualї
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_3/Cast 
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!while/lstm_cell_4/dropout_3/Mul_1И
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_3°
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mulІ
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_1І
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_2І
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   2)
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

:@@*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2)
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2D
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
Ђ
щ
-__inference_sequential_1_layer_call_fn_154240
conv1d_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	@А
	unknown_4:	 А
	unknown_5:	А
	unknown_6:	 А
	unknown_7:	А
	unknown_8:	@А
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1541762
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv1d_input
ђ
і
'__inference_lstm_3_layer_call_fn_155540

inputs
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1531862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
§
і
'__inference_lstm_4_layer_call_fn_156194

inputs
unknown:	 А
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1534362
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
з§
Э
!__inference__wrapped_model_151528
conv1d_inputU
?sequential_1_conv1d_conv1d_expanddims_1_readvariableop_resource: A
3sequential_1_conv1d_biasadd_readvariableop_resource: W
Asequential_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource: @C
5sequential_1_conv1d_1_biasadd_readvariableop_resource:@Q
>sequential_1_lstm_3_lstm_cell_3_matmul_readvariableop_resource:	@АS
@sequential_1_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource:	 АN
?sequential_1_lstm_3_lstm_cell_3_biasadd_readvariableop_resource:	АP
=sequential_1_lstm_4_lstm_cell_4_split_readvariableop_resource:	 АN
?sequential_1_lstm_4_lstm_cell_4_split_1_readvariableop_resource:	АJ
7sequential_1_lstm_4_lstm_cell_4_readvariableop_resource:	@АE
3sequential_1_dense_4_matmul_readvariableop_resource:@@B
4sequential_1_dense_4_biasadd_readvariableop_resource:@E
3sequential_1_dense_5_matmul_readvariableop_resource:@B
4sequential_1_dense_5_biasadd_readvariableop_resource:
identityИҐ*sequential_1/conv1d/BiasAdd/ReadVariableOpҐ6sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOpҐ,sequential_1/conv1d_1/BiasAdd/ReadVariableOpҐ8sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpҐ+sequential_1/dense_4/BiasAdd/ReadVariableOpҐ*sequential_1/dense_4/MatMul/ReadVariableOpҐ+sequential_1/dense_5/BiasAdd/ReadVariableOpҐ*sequential_1/dense_5/MatMul/ReadVariableOpҐ6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpҐ5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOpҐ7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpҐsequential_1/lstm_3/whileҐ.sequential_1/lstm_4/lstm_cell_4/ReadVariableOpҐ0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_1Ґ0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_2Ґ0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_3Ґ4sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOpҐ6sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOpҐsequential_1/lstm_4/while°
)sequential_1/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2+
)sequential_1/conv1d/conv1d/ExpandDims/dimЎ
%sequential_1/conv1d/conv1d/ExpandDims
ExpandDimsconv1d_input2sequential_1/conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%sequential_1/conv1d/conv1d/ExpandDimsф
6sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp?sequential_1_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype028
6sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOpЬ
+sequential_1/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential_1/conv1d/conv1d/ExpandDims_1/dimЗ
'sequential_1/conv1d/conv1d/ExpandDims_1
ExpandDims>sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:04sequential_1/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2)
'sequential_1/conv1d/conv1d/ExpandDims_1З
sequential_1/conv1d/conv1dConv2D.sequential_1/conv1d/conv1d/ExpandDims:output:00sequential_1/conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
sequential_1/conv1d/conv1dќ
"sequential_1/conv1d/conv1d/SqueezeSqueeze#sequential_1/conv1d/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€2$
"sequential_1/conv1d/conv1d/Squeeze»
*sequential_1/conv1d/BiasAdd/ReadVariableOpReadVariableOp3sequential_1_conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*sequential_1/conv1d/BiasAdd/ReadVariableOp№
sequential_1/conv1d/BiasAddBiasAdd+sequential_1/conv1d/conv1d/Squeeze:output:02sequential_1/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
sequential_1/conv1d/BiasAddШ
sequential_1/conv1d/ReluRelu$sequential_1/conv1d/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
sequential_1/conv1d/Relu•
+sequential_1/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2-
+sequential_1/conv1d_1/conv1d/ExpandDims/dimш
'sequential_1/conv1d_1/conv1d/ExpandDims
ExpandDims&sequential_1/conv1d/Relu:activations:04sequential_1/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2)
'sequential_1/conv1d_1/conv1d/ExpandDimsъ
8sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpAsequential_1_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02:
8sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp†
-sequential_1/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2/
-sequential_1/conv1d_1/conv1d/ExpandDims_1/dimП
)sequential_1/conv1d_1/conv1d/ExpandDims_1
ExpandDims@sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:06sequential_1/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2+
)sequential_1/conv1d_1/conv1d/ExpandDims_1П
sequential_1/conv1d_1/conv1dConv2D0sequential_1/conv1d_1/conv1d/ExpandDims:output:02sequential_1/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@*
paddingVALID*
strides
2
sequential_1/conv1d_1/conv1d‘
$sequential_1/conv1d_1/conv1d/SqueezeSqueeze%sequential_1/conv1d_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@*
squeeze_dims

э€€€€€€€€2&
$sequential_1/conv1d_1/conv1d/Squeezeќ
,sequential_1/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_1/conv1d_1/BiasAdd/ReadVariableOpд
sequential_1/conv1d_1/BiasAddBiasAdd-sequential_1/conv1d_1/conv1d/Squeeze:output:04sequential_1/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
sequential_1/conv1d_1/BiasAddЮ
sequential_1/conv1d_1/ReluRelu&sequential_1/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
sequential_1/conv1d_1/ReluШ
)sequential_1/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential_1/max_pooling1d/ExpandDims/dimф
%sequential_1/max_pooling1d/ExpandDims
ExpandDims(sequential_1/conv1d_1/Relu:activations:02sequential_1/max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@2'
%sequential_1/max_pooling1d/ExpandDimsр
"sequential_1/max_pooling1d/MaxPoolMaxPool.sequential_1/max_pooling1d/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2$
"sequential_1/max_pooling1d/MaxPoolЌ
"sequential_1/max_pooling1d/SqueezeSqueeze+sequential_1/max_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2$
"sequential_1/max_pooling1d/SqueezeС
sequential_1/lstm_3/ShapeShape+sequential_1/max_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/ShapeЬ
'sequential_1/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_3/strided_slice/stack†
)sequential_1/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_1†
)sequential_1/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_3/strided_slice/stack_2Џ
!sequential_1/lstm_3/strided_sliceStridedSlice"sequential_1/lstm_3/Shape:output:00sequential_1/lstm_3/strided_slice/stack:output:02sequential_1/lstm_3/strided_slice/stack_1:output:02sequential_1/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_3/strided_sliceД
sequential_1/lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential_1/lstm_3/zeros/mul/yЉ
sequential_1/lstm_3/zeros/mulMul*sequential_1/lstm_3/strided_slice:output:0(sequential_1/lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_3/zeros/mulЗ
 sequential_1/lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 sequential_1/lstm_3/zeros/Less/yЈ
sequential_1/lstm_3/zeros/LessLess!sequential_1/lstm_3/zeros/mul:z:0)sequential_1/lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_3/zeros/LessК
"sequential_1/lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential_1/lstm_3/zeros/packed/1”
 sequential_1/lstm_3/zeros/packedPack*sequential_1/lstm_3/strided_slice:output:0+sequential_1/lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_3/zeros/packedЗ
sequential_1/lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_3/zeros/Const≈
sequential_1/lstm_3/zerosFill)sequential_1/lstm_3/zeros/packed:output:0(sequential_1/lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_1/lstm_3/zerosИ
!sequential_1/lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential_1/lstm_3/zeros_1/mul/y¬
sequential_1/lstm_3/zeros_1/mulMul*sequential_1/lstm_3/strided_slice:output:0*sequential_1/lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_3/zeros_1/mulЛ
"sequential_1/lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2$
"sequential_1/lstm_3/zeros_1/Less/yњ
 sequential_1/lstm_3/zeros_1/LessLess#sequential_1/lstm_3/zeros_1/mul:z:0+sequential_1/lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_3/zeros_1/LessО
$sequential_1/lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2&
$sequential_1/lstm_3/zeros_1/packed/1ў
"sequential_1/lstm_3/zeros_1/packedPack*sequential_1/lstm_3/strided_slice:output:0-sequential_1/lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_3/zeros_1/packedЛ
!sequential_1/lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_3/zeros_1/ConstЌ
sequential_1/lstm_3/zeros_1Fill+sequential_1/lstm_3/zeros_1/packed:output:0*sequential_1/lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
sequential_1/lstm_3/zeros_1Э
"sequential_1/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_3/transpose/permџ
sequential_1/lstm_3/transpose	Transpose+sequential_1/max_pooling1d/Squeeze:output:0+sequential_1/lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_3/transposeЛ
sequential_1/lstm_3/Shape_1Shape!sequential_1/lstm_3/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_3/Shape_1†
)sequential_1/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_1/stack§
+sequential_1/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_1§
+sequential_1/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_1/stack_2ж
#sequential_1/lstm_3/strided_slice_1StridedSlice$sequential_1/lstm_3/Shape_1:output:02sequential_1/lstm_3/strided_slice_1/stack:output:04sequential_1/lstm_3/strided_slice_1/stack_1:output:04sequential_1/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_1≠
/sequential_1/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€21
/sequential_1/lstm_3/TensorArrayV2/element_shapeВ
!sequential_1/lstm_3/TensorArrayV2TensorListReserve8sequential_1/lstm_3/TensorArrayV2/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_3/TensorArrayV2з
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2K
Isequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape»
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_3/transpose:y:0Rsequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor†
)sequential_1/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_3/strided_slice_2/stack§
+sequential_1/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_1§
+sequential_1/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_2/stack_2ф
#sequential_1/lstm_3/strided_slice_2StridedSlice!sequential_1/lstm_3/transpose:y:02sequential_1/lstm_3/strided_slice_2/stack:output:04sequential_1/lstm_3/strided_slice_2/stack_1:output:04sequential_1/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_2о
5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp>sequential_1_lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype027
5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOpъ
&sequential_1/lstm_3/lstm_cell_3/MatMulMatMul,sequential_1/lstm_3/strided_slice_2:output:0=sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2(
&sequential_1/lstm_3/lstm_cell_3/MatMulф
7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp@sequential_1_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype029
7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpц
(sequential_1/lstm_3/lstm_cell_3/MatMul_1MatMul"sequential_1/lstm_3/zeros:output:0?sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2*
(sequential_1/lstm_3/lstm_cell_3/MatMul_1м
#sequential_1/lstm_3/lstm_cell_3/addAddV20sequential_1/lstm_3/lstm_cell_3/MatMul:product:02sequential_1/lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2%
#sequential_1/lstm_3/lstm_cell_3/addн
6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpщ
'sequential_1/lstm_3/lstm_cell_3/BiasAddBiasAdd'sequential_1/lstm_3/lstm_cell_3/add:z:0>sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2)
'sequential_1/lstm_3/lstm_cell_3/BiasAdd§
/sequential_1/lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_3/lstm_cell_3/split/split_dimњ
%sequential_1/lstm_3/lstm_cell_3/splitSplit8sequential_1/lstm_3/lstm_cell_3/split/split_dim:output:00sequential_1/lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2'
%sequential_1/lstm_3/lstm_cell_3/splitњ
'sequential_1/lstm_3/lstm_cell_3/SigmoidSigmoid.sequential_1/lstm_3/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2)
'sequential_1/lstm_3/lstm_cell_3/Sigmoid√
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_1Sigmoid.sequential_1/lstm_3/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_1Ў
#sequential_1/lstm_3/lstm_cell_3/mulMul-sequential_1/lstm_3/lstm_cell_3/Sigmoid_1:y:0$sequential_1/lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2%
#sequential_1/lstm_3/lstm_cell_3/mulґ
$sequential_1/lstm_3/lstm_cell_3/ReluRelu.sequential_1/lstm_3/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2&
$sequential_1/lstm_3/lstm_cell_3/Reluи
%sequential_1/lstm_3/lstm_cell_3/mul_1Mul+sequential_1/lstm_3/lstm_cell_3/Sigmoid:y:02sequential_1/lstm_3/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/lstm_3/lstm_cell_3/mul_1Ё
%sequential_1/lstm_3/lstm_cell_3/add_1AddV2'sequential_1/lstm_3/lstm_cell_3/mul:z:0)sequential_1/lstm_3/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/lstm_3/lstm_cell_3/add_1√
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_2Sigmoid.sequential_1/lstm_3/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2+
)sequential_1/lstm_3/lstm_cell_3/Sigmoid_2µ
&sequential_1/lstm_3/lstm_cell_3/Relu_1Relu)sequential_1/lstm_3/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2(
&sequential_1/lstm_3/lstm_cell_3/Relu_1м
%sequential_1/lstm_3/lstm_cell_3/mul_2Mul-sequential_1/lstm_3/lstm_cell_3/Sigmoid_2:y:04sequential_1/lstm_3/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2'
%sequential_1/lstm_3/lstm_cell_3/mul_2Ј
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    23
1sequential_1/lstm_3/TensorArrayV2_1/element_shapeИ
#sequential_1/lstm_3/TensorArrayV2_1TensorListReserve:sequential_1/lstm_3/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_3/TensorArrayV2_1v
sequential_1/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_3/timeІ
,sequential_1/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,sequential_1/lstm_3/while/maximum_iterationsТ
&sequential_1/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_3/while/loop_counterі
sequential_1/lstm_3/whileWhile/sequential_1/lstm_3/while/loop_counter:output:05sequential_1/lstm_3/while/maximum_iterations:output:0!sequential_1/lstm_3/time:output:0,sequential_1/lstm_3/TensorArrayV2_1:handle:0"sequential_1/lstm_3/zeros:output:0$sequential_1/lstm_3/zeros_1:output:0,sequential_1/lstm_3/strided_slice_1:output:0Ksequential_1/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_1_lstm_3_lstm_cell_3_matmul_readvariableop_resource@sequential_1_lstm_3_lstm_cell_3_matmul_1_readvariableop_resource?sequential_1_lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
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
%sequential_1_lstm_3_while_body_151189*1
cond)R'
%sequential_1_lstm_3_while_cond_151188*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
sequential_1/lstm_3/whileЁ
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2F
Dsequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeЄ
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_3/while:output:3Msequential_1/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype028
6sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack©
)sequential_1/lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2+
)sequential_1/lstm_3/strided_slice_3/stack§
+sequential_1/lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_3/strided_slice_3/stack_1§
+sequential_1/lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_3/strided_slice_3/stack_2Т
#sequential_1/lstm_3/strided_slice_3StridedSlice?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_3/strided_slice_3/stack:output:04sequential_1/lstm_3/strided_slice_3/stack_1:output:04sequential_1/lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2%
#sequential_1/lstm_3/strided_slice_3°
$sequential_1/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_3/transpose_1/permх
sequential_1/lstm_3/transpose_1	Transpose?sequential_1/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2!
sequential_1/lstm_3/transpose_1О
sequential_1/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_3/runtimeЙ
sequential_1/lstm_4/ShapeShape#sequential_1/lstm_3/transpose_1:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_4/ShapeЬ
'sequential_1/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_1/lstm_4/strided_slice/stack†
)sequential_1/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_4/strided_slice/stack_1†
)sequential_1/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_1/lstm_4/strided_slice/stack_2Џ
!sequential_1/lstm_4/strided_sliceStridedSlice"sequential_1/lstm_4/Shape:output:00sequential_1/lstm_4/strided_slice/stack:output:02sequential_1/lstm_4/strided_slice/stack_1:output:02sequential_1/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_1/lstm_4/strided_sliceД
sequential_1/lstm_4/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2!
sequential_1/lstm_4/zeros/mul/yЉ
sequential_1/lstm_4/zeros/mulMul*sequential_1/lstm_4/strided_slice:output:0(sequential_1/lstm_4/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_1/lstm_4/zeros/mulЗ
 sequential_1/lstm_4/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2"
 sequential_1/lstm_4/zeros/Less/yЈ
sequential_1/lstm_4/zeros/LessLess!sequential_1/lstm_4/zeros/mul:z:0)sequential_1/lstm_4/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_1/lstm_4/zeros/LessК
"sequential_1/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2$
"sequential_1/lstm_4/zeros/packed/1”
 sequential_1/lstm_4/zeros/packedPack*sequential_1/lstm_4/strided_slice:output:0+sequential_1/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_1/lstm_4/zeros/packedЗ
sequential_1/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_1/lstm_4/zeros/Const≈
sequential_1/lstm_4/zerosFill)sequential_1/lstm_4/zeros/packed:output:0(sequential_1/lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_4/zerosИ
!sequential_1/lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2#
!sequential_1/lstm_4/zeros_1/mul/y¬
sequential_1/lstm_4/zeros_1/mulMul*sequential_1/lstm_4/strided_slice:output:0*sequential_1/lstm_4/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_1/lstm_4/zeros_1/mulЛ
"sequential_1/lstm_4/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2$
"sequential_1/lstm_4/zeros_1/Less/yњ
 sequential_1/lstm_4/zeros_1/LessLess#sequential_1/lstm_4/zeros_1/mul:z:0+sequential_1/lstm_4/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_1/lstm_4/zeros_1/LessО
$sequential_1/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2&
$sequential_1/lstm_4/zeros_1/packed/1ў
"sequential_1/lstm_4/zeros_1/packedPack*sequential_1/lstm_4/strided_slice:output:0-sequential_1/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_1/lstm_4/zeros_1/packedЛ
!sequential_1/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_1/lstm_4/zeros_1/ConstЌ
sequential_1/lstm_4/zeros_1Fill+sequential_1/lstm_4/zeros_1/packed:output:0*sequential_1/lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/lstm_4/zeros_1Э
"sequential_1/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_1/lstm_4/transpose/perm”
sequential_1/lstm_4/transpose	Transpose#sequential_1/lstm_3/transpose_1:y:0+sequential_1/lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
sequential_1/lstm_4/transposeЛ
sequential_1/lstm_4/Shape_1Shape!sequential_1/lstm_4/transpose:y:0*
T0*
_output_shapes
:2
sequential_1/lstm_4/Shape_1†
)sequential_1/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_4/strided_slice_1/stack§
+sequential_1/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_1/stack_1§
+sequential_1/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_1/stack_2ж
#sequential_1/lstm_4/strided_slice_1StridedSlice$sequential_1/lstm_4/Shape_1:output:02sequential_1/lstm_4/strided_slice_1/stack:output:04sequential_1/lstm_4/strided_slice_1/stack_1:output:04sequential_1/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_1/lstm_4/strided_slice_1≠
/sequential_1/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€21
/sequential_1/lstm_4/TensorArrayV2/element_shapeВ
!sequential_1/lstm_4/TensorArrayV2TensorListReserve8sequential_1/lstm_4/TensorArrayV2/element_shape:output:0,sequential_1/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_1/lstm_4/TensorArrayV2з
Isequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2K
Isequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape»
;sequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_1/lstm_4/transpose:y:0Rsequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor†
)sequential_1/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_1/lstm_4/strided_slice_2/stack§
+sequential_1/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_2/stack_1§
+sequential_1/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_2/stack_2ф
#sequential_1/lstm_4/strided_slice_2StridedSlice!sequential_1/lstm_4/transpose:y:02sequential_1/lstm_4/strided_slice_2/stack:output:04sequential_1/lstm_4/strided_slice_2/stack_1:output:04sequential_1/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2%
#sequential_1/lstm_4/strided_slice_2і
/sequential_1/lstm_4/lstm_cell_4/ones_like/ShapeShape"sequential_1/lstm_4/zeros:output:0*
T0*
_output_shapes
:21
/sequential_1/lstm_4/lstm_cell_4/ones_like/ShapeІ
/sequential_1/lstm_4/lstm_cell_4/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  А?21
/sequential_1/lstm_4/lstm_cell_4/ones_like/ConstД
)sequential_1/lstm_4/lstm_cell_4/ones_likeFill8sequential_1/lstm_4/lstm_cell_4/ones_like/Shape:output:08sequential_1/lstm_4/lstm_cell_4/ones_like/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/lstm_cell_4/ones_like§
/sequential_1/lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_1/lstm_4/lstm_cell_4/split/split_dimл
4sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp=sequential_1_lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype026
4sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOpІ
%sequential_1/lstm_4/lstm_cell_4/splitSplit8sequential_1/lstm_4/lstm_cell_4/split/split_dim:output:0<sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2'
%sequential_1/lstm_4/lstm_cell_4/splitк
&sequential_1/lstm_4/lstm_cell_4/MatMulMatMul,sequential_1/lstm_4/strided_slice_2:output:0.sequential_1/lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_1/lstm_4/lstm_cell_4/MatMulо
(sequential_1/lstm_4/lstm_cell_4/MatMul_1MatMul,sequential_1/lstm_4/strided_slice_2:output:0.sequential_1/lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_1о
(sequential_1/lstm_4/lstm_cell_4/MatMul_2MatMul,sequential_1/lstm_4/strided_slice_2:output:0.sequential_1/lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_2о
(sequential_1/lstm_4/lstm_cell_4/MatMul_3MatMul,sequential_1/lstm_4/strided_slice_2:output:0.sequential_1/lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_3®
1sequential_1/lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential_1/lstm_4/lstm_cell_4/split_1/split_dimн
6sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp?sequential_1_lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype028
6sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOpЯ
'sequential_1/lstm_4/lstm_cell_4/split_1Split:sequential_1/lstm_4/lstm_cell_4/split_1/split_dim:output:0>sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2)
'sequential_1/lstm_4/lstm_cell_4/split_1у
'sequential_1/lstm_4/lstm_cell_4/BiasAddBiasAdd0sequential_1/lstm_4/lstm_cell_4/MatMul:product:00sequential_1/lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'sequential_1/lstm_4/lstm_cell_4/BiasAddщ
)sequential_1/lstm_4/lstm_cell_4/BiasAdd_1BiasAdd2sequential_1/lstm_4/lstm_cell_4/MatMul_1:product:00sequential_1/lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/lstm_cell_4/BiasAdd_1щ
)sequential_1/lstm_4/lstm_cell_4/BiasAdd_2BiasAdd2sequential_1/lstm_4/lstm_cell_4/MatMul_2:product:00sequential_1/lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/lstm_cell_4/BiasAdd_2щ
)sequential_1/lstm_4/lstm_cell_4/BiasAdd_3BiasAdd2sequential_1/lstm_4/lstm_cell_4/MatMul_3:product:00sequential_1/lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/lstm_cell_4/BiasAdd_3џ
#sequential_1/lstm_4/lstm_cell_4/mulMul"sequential_1/lstm_4/zeros:output:02sequential_1/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2%
#sequential_1/lstm_4/lstm_cell_4/mulя
%sequential_1/lstm_4/lstm_cell_4/mul_1Mul"sequential_1/lstm_4/zeros:output:02sequential_1/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/mul_1я
%sequential_1/lstm_4/lstm_cell_4/mul_2Mul"sequential_1/lstm_4/zeros:output:02sequential_1/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/mul_2я
%sequential_1/lstm_4/lstm_cell_4/mul_3Mul"sequential_1/lstm_4/zeros:output:02sequential_1/lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/mul_3ў
.sequential_1/lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp7sequential_1_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype020
.sequential_1/lstm_4/lstm_cell_4/ReadVariableOpї
3sequential_1/lstm_4/lstm_cell_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential_1/lstm_4/lstm_cell_4/strided_slice/stackњ
5sequential_1/lstm_4/lstm_cell_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    @   27
5sequential_1/lstm_4/lstm_cell_4/strided_slice/stack_1њ
5sequential_1/lstm_4/lstm_cell_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential_1/lstm_4/lstm_cell_4/strided_slice/stack_2Љ
-sequential_1/lstm_4/lstm_cell_4/strided_sliceStridedSlice6sequential_1/lstm_4/lstm_cell_4/ReadVariableOp:value:0<sequential_1/lstm_4/lstm_cell_4/strided_slice/stack:output:0>sequential_1/lstm_4/lstm_cell_4/strided_slice/stack_1:output:0>sequential_1/lstm_4/lstm_cell_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask2/
-sequential_1/lstm_4/lstm_cell_4/strided_sliceс
(sequential_1/lstm_4/lstm_cell_4/MatMul_4MatMul'sequential_1/lstm_4/lstm_cell_4/mul:z:06sequential_1/lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_4л
#sequential_1/lstm_4/lstm_cell_4/addAddV20sequential_1/lstm_4/lstm_cell_4/BiasAdd:output:02sequential_1/lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2%
#sequential_1/lstm_4/lstm_cell_4/addЄ
'sequential_1/lstm_4/lstm_cell_4/SigmoidSigmoid'sequential_1/lstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2)
'sequential_1/lstm_4/lstm_cell_4/SigmoidЁ
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp7sequential_1_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_1њ
5sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   27
5sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack√
7sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   29
7sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack_1√
7sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack_2»
/sequential_1/lstm_4/lstm_cell_4/strided_slice_1StridedSlice8sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_1:value:0>sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack:output:0@sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack_1:output:0@sequential_1/lstm_4/lstm_cell_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask21
/sequential_1/lstm_4/lstm_cell_4/strided_slice_1х
(sequential_1/lstm_4/lstm_cell_4/MatMul_5MatMul)sequential_1/lstm_4/lstm_cell_4/mul_1:z:08sequential_1/lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_5с
%sequential_1/lstm_4/lstm_cell_4/add_1AddV22sequential_1/lstm_4/lstm_cell_4/BiasAdd_1:output:02sequential_1/lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/add_1Њ
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_1Sigmoid)sequential_1/lstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_1№
%sequential_1/lstm_4/lstm_cell_4/mul_4Mul-sequential_1/lstm_4/lstm_cell_4/Sigmoid_1:y:0$sequential_1/lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/mul_4Ё
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp7sequential_1_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_2њ
5sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   27
5sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack√
7sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   29
7sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack_1√
7sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack_2»
/sequential_1/lstm_4/lstm_cell_4/strided_slice_2StridedSlice8sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_2:value:0>sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack:output:0@sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack_1:output:0@sequential_1/lstm_4/lstm_cell_4/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask21
/sequential_1/lstm_4/lstm_cell_4/strided_slice_2х
(sequential_1/lstm_4/lstm_cell_4/MatMul_6MatMul)sequential_1/lstm_4/lstm_cell_4/mul_2:z:08sequential_1/lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_6с
%sequential_1/lstm_4/lstm_cell_4/add_2AddV22sequential_1/lstm_4/lstm_cell_4/BiasAdd_2:output:02sequential_1/lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/add_2±
$sequential_1/lstm_4/lstm_cell_4/ReluRelu)sequential_1/lstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2&
$sequential_1/lstm_4/lstm_cell_4/Reluи
%sequential_1/lstm_4/lstm_cell_4/mul_5Mul+sequential_1/lstm_4/lstm_cell_4/Sigmoid:y:02sequential_1/lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/mul_5я
%sequential_1/lstm_4/lstm_cell_4/add_3AddV2)sequential_1/lstm_4/lstm_cell_4/mul_4:z:0)sequential_1/lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/add_3Ё
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp7sequential_1_lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype022
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_3њ
5sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   27
5sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack√
7sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack_1√
7sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack_2»
/sequential_1/lstm_4/lstm_cell_4/strided_slice_3StridedSlice8sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_3:value:0>sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack:output:0@sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack_1:output:0@sequential_1/lstm_4/lstm_cell_4/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:@@*

begin_mask*
end_mask21
/sequential_1/lstm_4/lstm_cell_4/strided_slice_3х
(sequential_1/lstm_4/lstm_cell_4/MatMul_7MatMul)sequential_1/lstm_4/lstm_cell_4/mul_3:z:08sequential_1/lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
(sequential_1/lstm_4/lstm_cell_4/MatMul_7с
%sequential_1/lstm_4/lstm_cell_4/add_4AddV22sequential_1/lstm_4/lstm_cell_4/BiasAdd_3:output:02sequential_1/lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/add_4Њ
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_2Sigmoid)sequential_1/lstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2+
)sequential_1/lstm_4/lstm_cell_4/Sigmoid_2µ
&sequential_1/lstm_4/lstm_cell_4/Relu_1Relu)sequential_1/lstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
&sequential_1/lstm_4/lstm_cell_4/Relu_1м
%sequential_1/lstm_4/lstm_cell_4/mul_6Mul-sequential_1/lstm_4/lstm_cell_4/Sigmoid_2:y:04sequential_1/lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2'
%sequential_1/lstm_4/lstm_cell_4/mul_6Ј
1sequential_1/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   23
1sequential_1/lstm_4/TensorArrayV2_1/element_shapeИ
#sequential_1/lstm_4/TensorArrayV2_1TensorListReserve:sequential_1/lstm_4/TensorArrayV2_1/element_shape:output:0,sequential_1/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_1/lstm_4/TensorArrayV2_1v
sequential_1/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_1/lstm_4/timeІ
,sequential_1/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2.
,sequential_1/lstm_4/while/maximum_iterationsТ
&sequential_1/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_1/lstm_4/while/loop_counter™
sequential_1/lstm_4/whileWhile/sequential_1/lstm_4/while/loop_counter:output:05sequential_1/lstm_4/while/maximum_iterations:output:0!sequential_1/lstm_4/time:output:0,sequential_1/lstm_4/TensorArrayV2_1:handle:0"sequential_1/lstm_4/zeros:output:0$sequential_1/lstm_4/zeros_1:output:0,sequential_1/lstm_4/strided_slice_1:output:0Ksequential_1/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0=sequential_1_lstm_4_lstm_cell_4_split_readvariableop_resource?sequential_1_lstm_4_lstm_cell_4_split_1_readvariableop_resource7sequential_1_lstm_4_lstm_cell_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_1_lstm_4_while_body_151379*1
cond)R'
%sequential_1_lstm_4_while_cond_151378*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
sequential_1/lstm_4/whileЁ
Dsequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2F
Dsequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeЄ
6sequential_1/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_1/lstm_4/while:output:3Msequential_1/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
element_dtype028
6sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack©
)sequential_1/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2+
)sequential_1/lstm_4/strided_slice_3/stack§
+sequential_1/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_1/lstm_4/strided_slice_3/stack_1§
+sequential_1/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_1/lstm_4/strided_slice_3/stack_2Т
#sequential_1/lstm_4/strided_slice_3StridedSlice?sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_1/lstm_4/strided_slice_3/stack:output:04sequential_1/lstm_4/strided_slice_3/stack_1:output:04sequential_1/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2%
#sequential_1/lstm_4/strided_slice_3°
$sequential_1/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_1/lstm_4/transpose_1/permх
sequential_1/lstm_4/transpose_1	Transpose?sequential_1/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_1/lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2!
sequential_1/lstm_4/transpose_1О
sequential_1/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_1/lstm_4/runtimeћ
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02,
*sequential_1/dense_4/MatMul/ReadVariableOpЎ
sequential_1/dense_4/MatMulMatMul,sequential_1/lstm_4/strided_slice_3:output:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_4/MatMulЋ
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_1/dense_4/BiasAdd/ReadVariableOp’
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_4/BiasAddЧ
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_1/dense_4/Reluћ
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*sequential_1/dense_5/MatMul/ReadVariableOp”
sequential_1/dense_5/MatMulMatMul'sequential_1/dense_4/Relu:activations:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_5/MatMulЋ
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_1/dense_5/BiasAdd/ReadVariableOp’
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_1/dense_5/BiasAddС
sequential_1/reshape_2/ShapeShape%sequential_1/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
sequential_1/reshape_2/ShapeҐ
*sequential_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_1/reshape_2/strided_slice/stack¶
,sequential_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_2/strided_slice/stack_1¶
,sequential_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_1/reshape_2/strided_slice/stack_2м
$sequential_1/reshape_2/strided_sliceStridedSlice%sequential_1/reshape_2/Shape:output:03sequential_1/reshape_2/strided_slice/stack:output:05sequential_1/reshape_2/strided_slice/stack_1:output:05sequential_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_1/reshape_2/strided_sliceТ
&sequential_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/reshape_2/Reshape/shape/1Т
&sequential_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&sequential_1/reshape_2/Reshape/shape/2У
$sequential_1/reshape_2/Reshape/shapePack-sequential_1/reshape_2/strided_slice:output:0/sequential_1/reshape_2/Reshape/shape/1:output:0/sequential_1/reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/reshape_2/Reshape/shape„
sequential_1/reshape_2/ReshapeReshape%sequential_1/dense_5/BiasAdd:output:0-sequential_1/reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2 
sequential_1/reshape_2/ReshapeЖ
IdentityIdentity'sequential_1/reshape_2/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityс
NoOpNoOp+^sequential_1/conv1d/BiasAdd/ReadVariableOp7^sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp-^sequential_1/conv1d_1/BiasAdd/ReadVariableOp9^sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp,^sequential_1/dense_4/BiasAdd/ReadVariableOp+^sequential_1/dense_4/MatMul/ReadVariableOp,^sequential_1/dense_5/BiasAdd/ReadVariableOp+^sequential_1/dense_5/MatMul/ReadVariableOp7^sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp6^sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOp8^sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^sequential_1/lstm_3/while/^sequential_1/lstm_4/lstm_cell_4/ReadVariableOp1^sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_11^sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_21^sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_35^sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOp7^sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOp^sequential_1/lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2X
*sequential_1/conv1d/BiasAdd/ReadVariableOp*sequential_1/conv1d/BiasAdd/ReadVariableOp2p
6sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp6sequential_1/conv1d/conv1d/ExpandDims_1/ReadVariableOp2\
,sequential_1/conv1d_1/BiasAdd/ReadVariableOp,sequential_1/conv1d_1/BiasAdd/ReadVariableOp2t
8sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp8sequential_1/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2Z
+sequential_1/dense_4/BiasAdd/ReadVariableOp+sequential_1/dense_4/BiasAdd/ReadVariableOp2X
*sequential_1/dense_4/MatMul/ReadVariableOp*sequential_1/dense_4/MatMul/ReadVariableOp2Z
+sequential_1/dense_5/BiasAdd/ReadVariableOp+sequential_1/dense_5/BiasAdd/ReadVariableOp2X
*sequential_1/dense_5/MatMul/ReadVariableOp*sequential_1/dense_5/MatMul/ReadVariableOp2p
6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp6sequential_1/lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2n
5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOp5sequential_1/lstm_3/lstm_cell_3/MatMul/ReadVariableOp2r
7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp7sequential_1/lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp26
sequential_1/lstm_3/whilesequential_1/lstm_3/while2`
.sequential_1/lstm_4/lstm_cell_4/ReadVariableOp.sequential_1/lstm_4/lstm_cell_4/ReadVariableOp2d
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_10sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_12d
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_20sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_22d
0sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_30sequential_1/lstm_4/lstm_cell_4/ReadVariableOp_32l
4sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOp4sequential_1/lstm_4/lstm_cell_4/split/ReadVariableOp2p
6sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOp6sequential_1/lstm_4/lstm_cell_4/split_1/ReadVariableOp26
sequential_1/lstm_4/whilesequential_1/lstm_4/while:Y U
+
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv1d_input
ОF
А
B__inference_lstm_3_layer_call_and_return_conditional_losses_151714

inputs%
lstm_cell_3_151632:	@А%
lstm_cell_3_151634:	 А!
lstm_cell_3_151636:	А
identityИҐ#lstm_cell_3/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_151632lstm_cell_3_151634lstm_cell_3_151636*
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1516312%
#lstm_cell_3/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_151632lstm_cell_3_151634lstm_cell_3_151636*
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
while_body_151645*
condR
while_cond_151644*K
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
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity|
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€@: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
Гv
ж
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_152543

inputs

states
states_10
split_readvariableop_resource:	 А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape”
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2бУ‘2&
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
:€€€€€€€€€@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/ShapeЎ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2•∞=2(
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
:€€€€€€€€€@2
dropout_1/GreaterEqualЕ
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_1/CastВ
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shapeў
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ьую2(
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
:€€€€€€€€€@2
dropout_2/GreaterEqualЕ
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_2/CastВ
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
dropout_3/Muld
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_3/Shapeў
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2У≠Е2(
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
:€€€€€€€€€@2
dropout_3/GreaterEqualЕ
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
dropout_3/CastВ
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dropout_3/Mul_1d
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_3^
mulMulstatesdropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
muld
mul_1Mulstatesdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1d
mul_2Mulstatesdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2
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

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2
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

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

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
?:€€€€€€€€€ :€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates
В»
Н
lstm_4_while_body_155226*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	 АI
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	АE
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:	@А
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:	 АG
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	АC
0lstm_4_while_lstm_cell_4_readvariableop_resource:	@АИҐ'lstm_4/while/lstm_cell_4/ReadVariableOpҐ)lstm_4/while/lstm_cell_4/ReadVariableOp_1Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_2Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_3Ґ-lstm_4/while/lstm_cell_4/split/ReadVariableOpҐ/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp—
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2@
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeэ
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2$
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
:€€€€€€€€€@2&
$lstm_4/while/lstm_cell_4/dropout/MulЂ
&lstm_4/while/lstm_cell_4/dropout/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2(
&lstm_4/while/lstm_cell_4/dropout/ShapeЮ
=lstm_4/while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform/lstm_4/while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2∞гИ2?
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
:€€€€€€€€€@2/
-lstm_4/while/lstm_cell_4/dropout/GreaterEqual 
%lstm_4/while/lstm_cell_4/dropout/CastCast1lstm_4/while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2'
%lstm_4/while/lstm_cell_4/dropout/Castё
&lstm_4/while/lstm_cell_4/dropout/Mul_1Mul(lstm_4/while/lstm_cell_4/dropout/Mul:z:0)lstm_4/while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2(
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
:€€€€€€€€€@2(
&lstm_4/while/lstm_cell_4/dropout_1/Mulѓ
(lstm_4/while/lstm_cell_4/dropout_1/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/dropout_1/Shape£
?lstm_4/while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ќїo2A
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
:€€€€€€€€€@21
/lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual–
'lstm_4/while/lstm_cell_4/dropout_1/CastCast3lstm_4/while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2)
'lstm_4/while/lstm_cell_4/dropout_1/Castж
(lstm_4/while/lstm_cell_4/dropout_1/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_1/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
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
:€€€€€€€€€@2(
&lstm_4/while/lstm_cell_4/dropout_2/Mulѓ
(lstm_4/while/lstm_cell_4/dropout_2/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/dropout_2/Shape§
?lstm_4/while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2яЩх2A
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
:€€€€€€€€€@21
/lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual–
'lstm_4/while/lstm_cell_4/dropout_2/CastCast3lstm_4/while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2)
'lstm_4/while/lstm_cell_4/dropout_2/Castж
(lstm_4/while/lstm_cell_4/dropout_2/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_2/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
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
:€€€€€€€€€@2(
&lstm_4/while/lstm_cell_4/dropout_3/Mulѓ
(lstm_4/while/lstm_cell_4/dropout_3/ShapeShape+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2*
(lstm_4/while/lstm_cell_4/dropout_3/Shape§
?lstm_4/while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform1lstm_4/while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ґбъ2A
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
:€€€€€€€€€@21
/lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual–
'lstm_4/while/lstm_cell_4/dropout_3/CastCast3lstm_4/while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2)
'lstm_4/while/lstm_cell_4/dropout_3/Castж
(lstm_4/while/lstm_cell_4/dropout_3/Mul_1Mul*lstm_4/while/lstm_cell_4/dropout_3/Mul:z:0+lstm_4/while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2*
(lstm_4/while/lstm_cell_4/dropout_3/Mul_1Ц
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_4/while/lstm_cell_4/split/split_dimЎ
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_4/while/lstm_cell_4/split/ReadVariableOpЛ
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2 
lstm_4/while/lstm_cell_4/splitа
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
lstm_4/while/lstm_cell_4/MatMulд
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_1д
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_2д
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_3Ъ
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_4/while/lstm_cell_4/split_1/split_dimЏ
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpГ
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2"
 lstm_4/while/lstm_cell_4/split_1„
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2"
 lstm_4/while/lstm_cell_4/BiasAddЁ
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/BiasAdd_1Ё
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/BiasAdd_2Ё
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/BiasAdd_3љ
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2*lstm_4/while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/while/lstm_cell_4/mul√
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_1√
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_2√
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2,lstm_4/while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_3∆
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   20
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

:@@*

begin_mask*
end_mask2(
&lstm_4/while/lstm_cell_4/strided_slice’
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_4ѕ
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/while/lstm_cell_4/add£
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2"
 lstm_4/while/lstm_cell_4/Sigmoid 
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_1±
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_4/while/lstm_cell_4/strided_slice_1/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   22
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

:@@*

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_1ў
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_5’
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_1©
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/Sigmoid_1љ
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_4 
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_2±
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   20
.lstm_4/while/lstm_cell_4/strided_slice_2/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   22
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

:@@*

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_2ў
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_6’
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_2Ь
lstm_4/while/lstm_cell_4/ReluRelu"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/while/lstm_cell_4/Reluћ
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_5√
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_3 
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_3±
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   20
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

:@@*

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_3ў
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_7’
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_4©
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/Sigmoid_2†
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
lstm_4/while/lstm_cell_4/Relu_1–
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
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
:€€€€€€€€€@2
lstm_4/while/Identity_4®
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2R
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
ђ
і
'__inference_lstm_3_layer_call_fn_155551

inputs
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1540552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
®

ѕ
lstm_4_while_cond_154747*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_154747___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_154747___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_154747___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_154747___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ућ
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_157305

inputs<
)lstm_cell_4_split_readvariableop_resource:	 А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
:€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout/MulД
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeч
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2фсґ22
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
:€€€€€€€€€@2"
 lstm_cell_4/dropout/GreaterEqual£
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout/Cast™
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_1/MulИ
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shapeэ
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ЙЃН24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_1/GreaterEqual©
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_1/Cast≤
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_2/MulИ
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeэ
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2јв≤24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_2/GreaterEqual©
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_2/Cast≤
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_3/MulИ
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeэ
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2”µЕ24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_3/GreaterEqual©
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_3/Cast≤
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_3К
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulР
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Р
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Р
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_157140*
condR
while_cond_157139*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@*
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
:€€€€€€€€€@2
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
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2z
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
:€€€€€€€€€ 
 
_user_specified_nameinputs
й=
н
H__inference_sequential_1_layer_call_and_return_conditional_losses_154176

inputs#
conv1d_154127: 
conv1d_154129: %
conv1d_1_154132: @
conv1d_1_154134:@ 
lstm_3_154138:	@А 
lstm_3_154140:	 А
lstm_3_154142:	А 
lstm_4_154145:	 А
lstm_4_154147:	А 
lstm_4_154149:	@А 
dense_4_154152:@@
dense_4_154154:@ 
dense_5_154157:@
dense_5_154159:
identityИҐconv1d/StatefulPartitionedCallҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpҐ conv1d_1/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpО
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_154127conv1d_154129*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1529992 
conv1d/StatefulPartitionedCallє
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_154132conv1d_1_154134*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1530212"
 conv1d_1/StatefulPartitionedCallК
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_1530342
max_pooling1d/PartitionedCallњ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0lstm_3_154138lstm_3_154140lstm_3_154142*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1540552 
lstm_3/StatefulPartitionedCallЉ
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_154145lstm_4_154147lstm_4_154149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1538822 
lstm_4/StatefulPartitionedCall∞
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_154152dense_4_154154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1534552!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_154157dense_5_154159*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_1534712!
dense_5/StatefulPartitionedCallэ
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
E__inference_reshape_2_layer_call_and_return_conditional_losses_1534902
reshape_2/PartitionedCallі
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_154127*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_154145*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_2/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityИ
NoOpNoOp^conv1d/StatefulPartitionedCall0^conv1d/kernel/Regularizer/Square/ReadVariableOp!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Я
J
.__inference_max_pooling1d_layer_call_fn_155486

inputs
identityЁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_1515402
PartitionedCallВ
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_153302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_153302___redundant_placeholder04
0while_while_cond_153302___redundant_placeholder14
0while_while_cond_153302___redundant_placeholder24
0while_while_cond_153302___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
У[
Ф
B__inference_lstm_3_layer_call_and_return_conditional_losses_156004

inputs=
*lstm_cell_3_matmul_readvariableop_resource:	@А?
,lstm_cell_3_matmul_1_readvariableop_resource:	 А:
+lstm_cell_3_biasadd_readvariableop_resource:	А
identityИҐ"lstm_cell_3/BiasAdd/ReadVariableOpҐ!lstm_cell_3/MatMul/ReadVariableOpҐ#lstm_cell_3/MatMul_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp™
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMulЄ
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¶
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMul_1Ь
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/BiasAdd|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimп
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_cell_3/splitГ
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/SigmoidЗ
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_1И
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/ReluШ
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_1Н
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/add_1З
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Relu_1Ь
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_2П
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
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
while_body_155920*
condR
while_cond_155919*K
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
:€€€€€€€€€ *
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
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

Identity≈
NoOpNoOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€@: : : 2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
’
√
while_cond_151854
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_151854___redundant_placeholder04
0while_while_cond_151854___redundant_placeholder14
0while_while_cond_151854___redundant_placeholder24
0while_while_cond_151854___redundant_placeholder3
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
Щ
у
-__inference_sequential_1_layer_call_fn_154430

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	@А
	unknown_4:	 А
	unknown_5:	А
	unknown_6:	 А
	unknown_7:	А
	unknown_8:	@А
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:
identityИҐStatefulPartitionedCallЫ
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1535052
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
’
√
while_cond_156070
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_156070___redundant_placeholder04
0while_while_cond_156070___redundant_placeholder14
0while_while_cond_156070___redundant_placeholder24
0while_while_cond_156070___redundant_placeholder3
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
'__inference_lstm_4_layer_call_fn_156205

inputs
unknown:	 А
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1538822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_151540

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
÷
J
.__inference_max_pooling1d_layer_call_fn_155491

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_1530342
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
@:S O
+
_output_shapes
:€€€€€€€€€
@
 
_user_specified_nameinputs
ОF
А
B__inference_lstm_3_layer_call_and_return_conditional_losses_151924

inputs%
lstm_cell_3_151842:	@А%
lstm_cell_3_151844:	 А!
lstm_cell_3_151846:	А
identityИҐ#lstm_cell_3/StatefulPartitionedCallҐwhileD
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
 :€€€€€€€€€€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_3_151842lstm_cell_3_151844lstm_cell_3_151846*
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1517772%
#lstm_cell_3/StatefulPartitionedCallП
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_3_151842lstm_cell_3_151844lstm_cell_3_151846*
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
while_body_151855*
condR
while_cond_151854*K
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
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identity|
NoOpNoOp$^lstm_cell_3/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€@: : : 2J
#lstm_cell_3/StatefulPartitionedCall#lstm_cell_3/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
і
х
,__inference_lstm_cell_4_layer_call_fn_157511

inputs
states_0
states_1
unknown:	 А
	unknown_0:	А
	unknown_1:	@А
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
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1525432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

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
?:€€€€€€€€€ :€€€€€€€€€@:€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
ѕР
Н
lstm_4_while_body_154748*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0K
8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0:	 АI
:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0:	АE
2lstm_4_while_lstm_cell_4_readvariableop_resource_0:	@А
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorI
6lstm_4_while_lstm_cell_4_split_readvariableop_resource:	 АG
8lstm_4_while_lstm_cell_4_split_1_readvariableop_resource:	АC
0lstm_4_while_lstm_cell_4_readvariableop_resource:	@АИҐ'lstm_4/while/lstm_cell_4/ReadVariableOpҐ)lstm_4/while/lstm_cell_4/ReadVariableOp_1Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_2Ґ)lstm_4/while/lstm_cell_4/ReadVariableOp_3Ґ-lstm_4/while/lstm_cell_4/split/ReadVariableOpҐ/lstm_4/while/lstm_cell_4/split_1/ReadVariableOp—
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2@
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeэ
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/ones_likeЦ
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_4/while/lstm_cell_4/split/split_dimЎ
-lstm_4/while/lstm_cell_4/split/ReadVariableOpReadVariableOp8lstm_4_while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02/
-lstm_4/while/lstm_cell_4/split/ReadVariableOpЛ
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:05lstm_4/while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2 
lstm_4/while/lstm_cell_4/splitа
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
lstm_4/while/lstm_cell_4/MatMulд
!lstm_4/while/lstm_cell_4/MatMul_1MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_1д
!lstm_4/while/lstm_cell_4/MatMul_2MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_2д
!lstm_4/while/lstm_cell_4/MatMul_3MatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_3Ъ
*lstm_4/while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*lstm_4/while/lstm_cell_4/split_1/split_dimЏ
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_4/while/lstm_cell_4/split_1/ReadVariableOpГ
 lstm_4/while/lstm_cell_4/split_1Split3lstm_4/while/lstm_cell_4/split_1/split_dim:output:07lstm_4/while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2"
 lstm_4/while/lstm_cell_4/split_1„
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd)lstm_4/while/lstm_cell_4/MatMul:product:0)lstm_4/while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2"
 lstm_4/while/lstm_cell_4/BiasAddЁ
"lstm_4/while/lstm_cell_4/BiasAdd_1BiasAdd+lstm_4/while/lstm_cell_4/MatMul_1:product:0)lstm_4/while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/BiasAdd_1Ё
"lstm_4/while/lstm_cell_4/BiasAdd_2BiasAdd+lstm_4/while/lstm_cell_4/MatMul_2:product:0)lstm_4/while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/BiasAdd_2Ё
"lstm_4/while/lstm_cell_4/BiasAdd_3BiasAdd+lstm_4/while/lstm_cell_4/MatMul_3:product:0)lstm_4/while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/BiasAdd_3Њ
lstm_4/while/lstm_cell_4/mulMullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/while/lstm_cell_4/mul¬
lstm_4/while/lstm_cell_4/mul_1Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_1¬
lstm_4/while/lstm_cell_4/mul_2Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_2¬
lstm_4/while/lstm_cell_4/mul_3Mullstm_4_while_placeholder_2+lstm_4/while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_3∆
'lstm_4/while/lstm_cell_4/ReadVariableOpReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   20
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

:@@*

begin_mask*
end_mask2(
&lstm_4/while/lstm_cell_4/strided_slice’
!lstm_4/while/lstm_cell_4/MatMul_4MatMul lstm_4/while/lstm_cell_4/mul:z:0/lstm_4/while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_4ѕ
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/BiasAdd:output:0+lstm_4/while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/while/lstm_cell_4/add£
 lstm_4/while/lstm_cell_4/SigmoidSigmoid lstm_4/while/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2"
 lstm_4/while/lstm_cell_4/Sigmoid 
)lstm_4/while/lstm_cell_4/ReadVariableOp_1ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_1±
.lstm_4/while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   20
.lstm_4/while/lstm_cell_4/strided_slice_1/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   22
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

:@@*

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_1ў
!lstm_4/while/lstm_cell_4/MatMul_5MatMul"lstm_4/while/lstm_cell_4/mul_1:z:01lstm_4/while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_5’
lstm_4/while/lstm_cell_4/add_1AddV2+lstm_4/while/lstm_cell_4/BiasAdd_1:output:0+lstm_4/while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_1©
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/Sigmoid_1љ
lstm_4/while/lstm_cell_4/mul_4Mul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_4 
)lstm_4/while/lstm_cell_4/ReadVariableOp_2ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_2±
.lstm_4/while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   20
.lstm_4/while/lstm_cell_4/strided_slice_2/stackµ
0lstm_4/while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   22
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

:@@*

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_2ў
!lstm_4/while/lstm_cell_4/MatMul_6MatMul"lstm_4/while/lstm_cell_4/mul_2:z:01lstm_4/while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_6’
lstm_4/while/lstm_cell_4/add_2AddV2+lstm_4/while/lstm_cell_4/BiasAdd_2:output:0+lstm_4/while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_2Ь
lstm_4/while/lstm_cell_4/ReluRelu"lstm_4/while/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/while/lstm_cell_4/Reluћ
lstm_4/while/lstm_cell_4/mul_5Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/mul_5√
lstm_4/while/lstm_cell_4/add_3AddV2"lstm_4/while/lstm_cell_4/mul_4:z:0"lstm_4/while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_3 
)lstm_4/while/lstm_cell_4/ReadVariableOp_3ReadVariableOp2lstm_4_while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02+
)lstm_4/while/lstm_cell_4/ReadVariableOp_3±
.lstm_4/while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   20
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

:@@*

begin_mask*
end_mask2*
(lstm_4/while/lstm_cell_4/strided_slice_3ў
!lstm_4/while/lstm_cell_4/MatMul_7MatMul"lstm_4/while/lstm_cell_4/mul_3:z:01lstm_4/while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/while/lstm_cell_4/MatMul_7’
lstm_4/while/lstm_cell_4/add_4AddV2+lstm_4/while/lstm_cell_4/BiasAdd_3:output:0+lstm_4/while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
lstm_4/while/lstm_cell_4/add_4©
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid"lstm_4/while/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/while/lstm_cell_4/Sigmoid_2†
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
lstm_4/while/lstm_cell_4/Relu_1–
lstm_4/while/lstm_cell_4/mul_6Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2 
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
:€€€€€€€€€@2
lstm_4/while/Identity_4®
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_3:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2R
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
У[
Ф
B__inference_lstm_3_layer_call_and_return_conditional_losses_153186

inputs=
*lstm_cell_3_matmul_readvariableop_resource:	@А?
,lstm_cell_3_matmul_1_readvariableop_resource:	 А:
+lstm_cell_3_biasadd_readvariableop_resource:	А
identityИҐ"lstm_cell_3/BiasAdd/ReadVariableOpҐ!lstm_cell_3/MatMul/ReadVariableOpҐ#lstm_cell_3/MatMul_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp™
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMulЄ
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¶
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMul_1Ь
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/BiasAdd|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimп
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_cell_3/splitГ
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/SigmoidЗ
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_1И
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/ReluШ
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_1Н
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/add_1З
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Relu_1Ь
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_2П
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
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
while_body_153102*
condR
while_cond_153101*K
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
:€€€€€€€€€ *
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
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

Identity≈
NoOpNoOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€@: : : 2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
£
e
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_153034

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
:€€€€€€€€€
@2

ExpandDimsЯ
MaxPoolMaxPoolExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2	
MaxPool|
SqueezeSqueezeMaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2	
Squeezeh
IdentityIdentitySqueeze:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€
@:S O
+
_output_shapes
:€€€€€€€€€
@
 
_user_specified_nameinputs
Н†
Э
B__inference_lstm_4_layer_call_and_return_conditional_losses_156448
inputs_0<
)lstm_cell_4_split_readvariableop_resource:	 А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
 :€€€€€€€€€€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
lstm_cell_4/ones_like|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_3Л
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulП
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1П
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_156315*
condR
while_cond_156314*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
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
:€€€€€€€€€@*
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
 :€€€€€€€€€€€€€€€€€€@2
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
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2z
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
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
ƒ~
Ш	
while_body_156315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	 АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	 А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	@АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2
while/lstm_cell_4/ones_likeИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_3Ґ
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul¶
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_1¶
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_2¶
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   2)
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

:@@*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2)
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2D
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
т
Г
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_151631

inputs

states
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
?:€€€€€€€€€@:€€€€€€€€€ :€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_namestates
®Ќ
Э
B__inference_lstm_4_layer_call_and_return_conditional_losses_156755
inputs_0<
)lstm_cell_4_split_readvariableop_resource:	 А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
 :€€€€€€€€€€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout/MulД
lstm_cell_4/dropout/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout/Shapeч
0lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform"lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2№“°22
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
:€€€€€€€€€@2"
 lstm_cell_4/dropout/GreaterEqual£
lstm_cell_4/dropout/CastCast$lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout/Cast™
lstm_cell_4/dropout/Mul_1Mullstm_cell_4/dropout/Mul:z:0lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_1/MulИ
lstm_cell_4/dropout_1/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_1/Shapeэ
2lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2±ћ≤24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_1/GreaterEqual©
lstm_cell_4/dropout_1/CastCast&lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_1/Cast≤
lstm_cell_4/dropout_1/Mul_1Mullstm_cell_4/dropout_1/Mul:z:0lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_2/MulИ
lstm_cell_4/dropout_2/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_2/Shapeь
2lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2€Ѓ24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_2/GreaterEqual©
lstm_cell_4/dropout_2/CastCast&lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_2/Cast≤
lstm_cell_4/dropout_2/Mul_1Mullstm_cell_4/dropout_2/Mul:z:0lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
lstm_cell_4/dropout_3/MulИ
lstm_cell_4/dropout_3/ShapeShapelstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2
lstm_cell_4/dropout_3/Shapeэ
2lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform$lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Ј©—24
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
:€€€€€€€€€@2$
"lstm_cell_4/dropout_3/GreaterEqual©
lstm_cell_4/dropout_3/CastCast&lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_3/Cast≤
lstm_cell_4/dropout_3/Mul_1Mullstm_cell_4/dropout_3/Mul:z:0lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/dropout_3/Mul_1|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_3К
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulР
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1Р
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2Р
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_156590*
condR
while_cond_156589*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
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
:€€€€€€€€€@*
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
 :€€€€€€€€€€€€€€€€€€@2
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
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2z
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
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
’
√
while_cond_156864
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_156864___redundant_placeholder04
0while_while_cond_156864___redundant_placeholder14
0while_while_cond_156864___redundant_placeholder24
0while_while_cond_156864___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
Ђ
щ
-__inference_sequential_1_layer_call_fn_153536
conv1d_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	@А
	unknown_4:	 А
	unknown_5:	А
	unknown_6:	 А
	unknown_7:	А
	unknown_8:	@А
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:
identityИҐStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1535052
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv1d_input
£R
ж
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_152310

inputs

states
states_10
split_readvariableop_resource:	 А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
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
:€€€€€€€€€@2
	ones_liked
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_3_
mulMulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mulc
mul_1Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1c
mul_2Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2c
mul_3Mulstatesones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2
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

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2
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

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

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
?:€€€€€€€€€ :€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates:OK
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_namestates
ƒ~
Ш	
while_body_153303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	 АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	 А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	@АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2
while/lstm_cell_4/ones_likeИ
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_3Ґ
while/lstm_cell_4/mulMulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul¶
while/lstm_cell_4/mul_1Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_1¶
while/lstm_cell_4/mul_2Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_2¶
while/lstm_cell_4/mul_3Mulwhile_placeholder_2$while/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   2)
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

:@@*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2)
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2D
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
¬>
«
while_body_153102
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_3_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_3_matmul_readvariableop_resource:	@АE
2while_lstm_cell_3_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ(while/lstm_cell_3/BiasAdd/ReadVariableOpҐ'while/lstm_cell_3/MatMul/ReadVariableOpҐ)while/lstm_cell_3/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem∆
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp‘
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMulћ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOpљ
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMul_1і
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/add≈
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpЅ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/BiasAddИ
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dimЗ
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
while/lstm_cell_3/splitХ
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/SigmoidЩ
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_1Э
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mulМ
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu∞
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_1•
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/add_1Щ
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_2Л
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu_1і
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_2я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5џ

while/NoOpNoOp)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*"
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
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
while_cond_156589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_156589___redundant_placeholder04
0while_while_cond_156589___redundant_placeholder14
0while_while_cond_156589___redundant_placeholder24
0while_while_cond_156589___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
≥R
и
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_157592

inputs
states_0
states_10
split_readvariableop_resource:	 А.
split_1_readvariableop_resource:	А*
readvariableop_resource:	@А
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
:€€€€€€€€€@2
	ones_liked
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimЛ
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02
split/ReadVariableOpІ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
splitd
MatMulMatMulinputssplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulh
MatMul_1MatMulinputssplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_1h
MatMul_2MatMulinputssplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_2h
MatMul_3MatMulinputssplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_3h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dimН
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:А*
dtype02
split_1/ReadVariableOpЯ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2	
split_1s
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddy
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_1y
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_2y
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
	BiasAdd_3a
mulMulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mule
mul_1Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_1e
mul_2Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_2e
mul_3Mulstates_0ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_3y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2
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

:@@*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_4k
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
Sigmoid}
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2
strided_slice_1/stackГ
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2
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

:@@*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_5q
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_1`
mul_4MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_4}
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2
strided_slice_2/stackГ
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_6q
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_2Q
ReluRelu	add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Reluh
mul_5MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_5_
add_3AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_3}
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes
:	@А*
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2
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

:@@*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMul	mul_3:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2

MatMul_7q
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
add_4^
	Sigmoid_2Sigmoid	add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
	Sigmoid_2U
Relu_1Relu	add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relu_1l
mul_6MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
mul_6ў
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muld
IdentityIdentity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh

Identity_1Identity	mul_6:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity_1h

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

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
?:€€€€€€€€€ :€€€€€€€€€@:€€€€€€€€€@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/0:QM
'
_output_shapes
:€€€€€€€€€@
"
_user_specified_name
states/1
’
√
while_cond_157139
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_157139___redundant_placeholder04
0while_while_cond_157139___redundant_placeholder14
0while_while_cond_157139___redundant_placeholder24
0while_while_cond_157139___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
ъ
Е
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_157439

inputs
states_0
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
?:€€€€€€€€€@:€€€€€€€€€ :€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
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
–Q
Њ
B__inference_lstm_4_layer_call_and_return_conditional_losses_152399

inputs%
lstm_cell_4_152311:	 А!
lstm_cell_4_152313:	А%
lstm_cell_4_152315:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
 :€€€€€€€€€€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_152311lstm_cell_4_152313lstm_cell_4_152315*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1523102%
#lstm_cell_4/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_152311lstm_cell_4_152313lstm_cell_4_152315*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_152324*
condR
while_cond_152323*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
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
:€€€€€€€€€@*
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
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeќ
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_4_152311*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityЇ
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ЅH
І

lstm_3_while_body_154558*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3)
%lstm_3_while_lstm_3_strided_slice_1_0e
alstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0:	@АN
;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АI
:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
lstm_3_while_identity
lstm_3_while_identity_1
lstm_3_while_identity_2
lstm_3_while_identity_3
lstm_3_while_identity_4
lstm_3_while_identity_5'
#lstm_3_while_lstm_3_strided_slice_1c
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensorJ
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource:	@АL
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource:	 АG
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpҐ.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpҐ0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp—
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2@
>lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeэ
0lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0lstm_3_while_placeholderGlstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype022
0lstm_3/while/TensorArrayV2Read/TensorListGetItemџ
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype020
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOpр
lstm_3/while/lstm_cell_3/MatMulMatMul7lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2!
lstm_3/while/lstm_cell_3/MatMulб
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype022
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOpў
!lstm_3/while/lstm_cell_3/MatMul_1MatMullstm_3_while_placeholder_28lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2#
!lstm_3/while/lstm_cell_3/MatMul_1–
lstm_3/while/lstm_cell_3/addAddV2)lstm_3/while/lstm_cell_3/MatMul:product:0+lstm_3/while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/while/lstm_cell_3/addЏ
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype021
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOpЁ
 lstm_3/while/lstm_cell_3/BiasAddBiasAdd lstm_3/while/lstm_cell_3/add:z:07lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2"
 lstm_3/while/lstm_cell_3/BiasAddЦ
(lstm_3/while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_3/while/lstm_cell_3/split/split_dim£
lstm_3/while/lstm_cell_3/splitSplit1lstm_3/while/lstm_cell_3/split/split_dim:output:0)lstm_3/while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2 
lstm_3/while/lstm_cell_3/split™
 lstm_3/while/lstm_cell_3/SigmoidSigmoid'lstm_3/while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2"
 lstm_3/while/lstm_cell_3/SigmoidЃ
"lstm_3/while/lstm_cell_3/Sigmoid_1Sigmoid'lstm_3/while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_3/while/lstm_cell_3/Sigmoid_1є
lstm_3/while/lstm_cell_3/mulMul&lstm_3/while/lstm_cell_3/Sigmoid_1:y:0lstm_3_while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/lstm_cell_3/mul°
lstm_3/while/lstm_cell_3/ReluRelu'lstm_3/while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/lstm_cell_3/Reluћ
lstm_3/while/lstm_cell_3/mul_1Mul$lstm_3/while/lstm_cell_3/Sigmoid:y:0+lstm_3/while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_3/while/lstm_cell_3/mul_1Ѕ
lstm_3/while/lstm_cell_3/add_1AddV2 lstm_3/while/lstm_cell_3/mul:z:0"lstm_3/while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_3/while/lstm_cell_3/add_1Ѓ
"lstm_3/while/lstm_cell_3/Sigmoid_2Sigmoid'lstm_3/while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2$
"lstm_3/while/lstm_cell_3/Sigmoid_2†
lstm_3/while/lstm_cell_3/Relu_1Relu"lstm_3/while/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2!
lstm_3/while/lstm_cell_3/Relu_1–
lstm_3/while/lstm_cell_3/mul_2Mul&lstm_3/while/lstm_cell_3/Sigmoid_2:y:0-lstm_3/while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2 
lstm_3/while/lstm_cell_3/mul_2В
1lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_3_while_placeholder_1lstm_3_while_placeholder"lstm_3/while/lstm_cell_3/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_3/while/TensorArrayV2Write/TensorListSetItemj
lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add/yЕ
lstm_3/while/addAddV2lstm_3_while_placeholderlstm_3/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/addn
lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_3/while/add_1/yЩ
lstm_3/while/add_1AddV2&lstm_3_while_lstm_3_while_loop_counterlstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_3/while/add_1З
lstm_3/while/IdentityIdentitylstm_3/while/add_1:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity°
lstm_3/while/Identity_1Identity,lstm_3_while_lstm_3_while_maximum_iterations^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_1Й
lstm_3/while/Identity_2Identitylstm_3/while/add:z:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_2ґ
lstm_3/while/Identity_3IdentityAlstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_3/while/NoOp*
T0*
_output_shapes
: 2
lstm_3/while/Identity_3®
lstm_3/while/Identity_4Identity"lstm_3/while/lstm_cell_3/mul_2:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/Identity_4®
lstm_3/while/Identity_5Identity"lstm_3/while/lstm_cell_3/add_1:z:0^lstm_3/while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/while/Identity_5ю
lstm_3/while/NoOpNoOp0^lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/^lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp1^lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_3/while/NoOp"7
lstm_3_while_identitylstm_3/while/Identity:output:0";
lstm_3_while_identity_1 lstm_3/while/Identity_1:output:0";
lstm_3_while_identity_2 lstm_3/while/Identity_2:output:0";
lstm_3_while_identity_3 lstm_3/while/Identity_3:output:0";
lstm_3_while_identity_4 lstm_3/while/Identity_4:output:0";
lstm_3_while_identity_5 lstm_3/while/Identity_5:output:0"L
#lstm_3_while_lstm_3_strided_slice_1%lstm_3_while_lstm_3_strided_slice_1_0"v
8lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource:lstm_3_while_lstm_cell_3_biasadd_readvariableop_resource_0"x
9lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource;lstm_3_while_lstm_cell_3_matmul_1_readvariableop_resource_0"t
7lstm_3_while_lstm_cell_3_matmul_readvariableop_resource9lstm_3_while_lstm_cell_3_matmul_readvariableop_resource_0"ƒ
_lstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensoralstm_3_while_tensorarrayv2read_tensorlistgetitem_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2b
/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp/lstm_3/while/lstm_cell_3/BiasAdd/ReadVariableOp2`
.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp.lstm_3/while/lstm_cell_3/MatMul/ReadVariableOp2d
0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp0lstm_3/while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
Ц
√
B__inference_conv1d_layer_call_and_return_conditional_losses_152999

inputsA
+conv1d_expanddims_1_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐ"conv1d/ExpandDims_1/ReadVariableOpҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/ExpandDims/dimЦ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1d/ExpandDimsЄ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЈ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1Ј
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv1dТ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€ 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
Relu“
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mulq
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ 2

IdentityЊ
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp0^conv1d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ы=
у
H__inference_sequential_1_layer_call_and_return_conditional_losses_154292
conv1d_input#
conv1d_154243: 
conv1d_154245: %
conv1d_1_154248: @
conv1d_1_154250:@ 
lstm_3_154254:	@А 
lstm_3_154256:	 А
lstm_3_154258:	А 
lstm_4_154261:	 А
lstm_4_154263:	А 
lstm_4_154265:	@А 
dense_4_154268:@@
dense_4_154270:@ 
dense_5_154273:@
dense_5_154275:
identityИҐconv1d/StatefulPartitionedCallҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpҐ conv1d_1/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpФ
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_154243conv1d_154245*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1529992 
conv1d/StatefulPartitionedCallє
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_154248conv1d_1_154250*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1530212"
 conv1d_1/StatefulPartitionedCallК
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_1530342
max_pooling1d/PartitionedCallњ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0lstm_3_154254lstm_3_154256lstm_3_154258*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1531862 
lstm_3/StatefulPartitionedCallЉ
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_154261lstm_4_154263lstm_4_154265*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1534362 
lstm_4/StatefulPartitionedCall∞
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_154268dense_4_154270*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1534552!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_154273dense_5_154275*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_1534712!
dense_5/StatefulPartitionedCallэ
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
E__inference_reshape_2_layer_call_and_return_conditional_losses_1534902
reshape_2/PartitionedCallі
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_154243*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_154261*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_2/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityИ
NoOpNoOp^conv1d/StatefulPartitionedCall0^conv1d/kernel/Regularizer/Square/ReadVariableOp!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:Y U
+
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv1d_input
„Я
Ы
B__inference_lstm_4_layer_call_and_return_conditional_losses_156998

inputs<
)lstm_cell_4_split_readvariableop_resource:	 А:
+lstm_cell_4_split_1_readvariableop_resource:	А6
#lstm_cell_4_readvariableop_resource:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
:€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
lstm_cell_4/ones_like|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dimѓ
 lstm_cell_4/split/ReadVariableOpReadVariableOp)lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02"
 lstm_cell_4/split/ReadVariableOp„
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0(lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_cell_4/splitЪ
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMulЮ
lstm_cell_4/MatMul_1MatMulstrided_slice_2:output:0lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_1Ю
lstm_cell_4/MatMul_2MatMulstrided_slice_2:output:0lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_2Ю
lstm_cell_4/MatMul_3MatMulstrided_slice_2:output:0lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_3А
lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell_4/split_1/split_dim±
"lstm_cell_4/split_1/ReadVariableOpReadVariableOp+lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_4/split_1/ReadVariableOpѕ
lstm_cell_4/split_1Split&lstm_cell_4/split_1/split_dim:output:0*lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_cell_4/split_1£
lstm_cell_4/BiasAddBiasAddlstm_cell_4/MatMul:product:0lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd©
lstm_cell_4/BiasAdd_1BiasAddlstm_cell_4/MatMul_1:product:0lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_1©
lstm_cell_4/BiasAdd_2BiasAddlstm_cell_4/MatMul_2:product:0lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_2©
lstm_cell_4/BiasAdd_3BiasAddlstm_cell_4/MatMul_3:product:0lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/BiasAdd_3Л
lstm_cell_4/mulMulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mulП
lstm_cell_4/mul_1Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_1П
lstm_cell_4/mul_2Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_2П
lstm_cell_4/mul_3Mulzeros:output:0lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_3Э
lstm_cell_4/ReadVariableOpReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice°
lstm_cell_4/MatMul_4MatMullstm_cell_4/mul:z:0"lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_4Ы
lstm_cell_4/addAddV2lstm_cell_4/BiasAdd:output:0lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add|
lstm_cell_4/SigmoidSigmoidlstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid°
lstm_cell_4/ReadVariableOp_1ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_1Ч
!lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2#
!lstm_cell_4/strided_slice_1/stackЫ
#lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_1•
lstm_cell_4/MatMul_5MatMullstm_cell_4/mul_1:z:0$lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_5°
lstm_cell_4/add_1AddV2lstm_cell_4/BiasAdd_1:output:0lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_1В
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_1М
lstm_cell_4/mul_4Mullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_4°
lstm_cell_4/ReadVariableOp_2ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_2Ч
!lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2#
!lstm_cell_4/strided_slice_2/stackЫ
#lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2%
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_2•
lstm_cell_4/MatMul_6MatMullstm_cell_4/mul_2:z:0$lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_6°
lstm_cell_4/add_2AddV2lstm_cell_4/BiasAdd_2:output:0lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_2u
lstm_cell_4/ReluRelulstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/ReluШ
lstm_cell_4/mul_5Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_5П
lstm_cell_4/add_3AddV2lstm_cell_4/mul_4:z:0lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_3°
lstm_cell_4/ReadVariableOp_3ReadVariableOp#lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02
lstm_cell_4/ReadVariableOp_3Ч
!lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2#
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

:@@*

begin_mask*
end_mask2
lstm_cell_4/strided_slice_3•
lstm_cell_4/MatMul_7MatMullstm_cell_4/mul_3:z:0$lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/MatMul_7°
lstm_cell_4/add_4AddV2lstm_cell_4/BiasAdd_3:output:0lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/add_4В
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/Relu_1Ь
lstm_cell_4/mul_6Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_cell_4/mul_6П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_156865*
condR
while_cond_156864*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeи
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@*
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
:€€€€€€€€€@2
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
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity÷
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp^lstm_cell_4/ReadVariableOp^lstm_cell_4/ReadVariableOp_1^lstm_cell_4/ReadVariableOp_2^lstm_cell_4/ReadVariableOp_3!^lstm_cell_4/split/ReadVariableOp#^lstm_cell_4/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€ : : : 2z
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
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ји
Б
H__inference_sequential_1_layer_call_and_return_conditional_losses_154909

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_1_biasadd_readvariableop_resource:@D
1lstm_3_lstm_cell_3_matmul_readvariableop_resource:	@АF
3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource:	 АA
2lstm_3_lstm_cell_3_biasadd_readvariableop_resource:	АC
0lstm_4_lstm_cell_4_split_readvariableop_resource:	 АA
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:	А=
*lstm_4_lstm_cell_4_readvariableop_resource:	@А8
&dense_4_matmul_readvariableop_resource:@@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:
identityИҐconv1d/BiasAdd/ReadVariableOpҐ)conv1d/conv1d/ExpandDims_1/ReadVariableOpҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpҐconv1d_1/BiasAdd/ReadVariableOpҐ+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐ)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpҐ(lstm_3/lstm_cell_3/MatMul/ReadVariableOpҐ*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpҐlstm_3/whileҐ!lstm_4/lstm_cell_4/ReadVariableOpҐ#lstm_4/lstm_cell_4/ReadVariableOp_1Ґ#lstm_4/lstm_cell_4/ReadVariableOp_2Ґ#lstm_4/lstm_cell_4/ReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐ'lstm_4/lstm_cell_4/split/ReadVariableOpҐ)lstm_4/lstm_cell_4/split_1/ReadVariableOpҐlstm_4/whileЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/conv1d/ExpandDims/dimЂ
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1d/conv1d/ExpandDimsЌ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim”
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/conv1d/ExpandDims_1”
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv1d/conv1dІ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€2
conv1d/conv1d/Squeeze°
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp®
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
conv1d/ReluЛ
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_1/conv1d/ExpandDims/dimƒ
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv1d_1/conv1d/ExpandDims”
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimџ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_1/conv1d/ExpandDims_1џ
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@*
paddingVALID*
strides
2
conv1d_1/conv1d≠
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@*
squeeze_dims

э€€€€€€€€2
conv1d_1/conv1d/SqueezeІ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp∞
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
conv1d_1/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dimј
max_pooling1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@2
max_pooling1d/ExpandDims…
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool¶
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
max_pooling1d/Squeezej
lstm_3/ShapeShapemax_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_3/ShapeВ
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stackЖ
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1Ж
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2М
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/mul/yИ
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros/Less/yГ
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/packed/1Я
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/ConstС
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/mul/yО
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros_1/Less/yЛ
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/packed/1•
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/ConstЩ
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/zeros_1Г
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/permІ
lstm_3/transpose	Transposemax_pooling1d/Squeeze:output:0lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1Ж
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stackК
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1К
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2Ш
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1У
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_3/TensorArrayV2/element_shapeќ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2Ќ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensorЖ
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stackК
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1К
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2¶
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_3/strided_slice_2«
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp∆
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/MatMulЌ
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¬
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/MatMul_1Є
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/MatMul:product:0%lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/add∆
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp≈
lstm_3/lstm_cell_3/BiasAddBiasAddlstm_3/lstm_cell_3/add:z:01lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/BiasAddК
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dimЛ
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0#lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_3/lstm_cell_3/splitШ
lstm_3/lstm_cell_3/SigmoidSigmoid!lstm_3/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/SigmoidЬ
lstm_3/lstm_cell_3/Sigmoid_1Sigmoid!lstm_3/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Sigmoid_1§
lstm_3/lstm_cell_3/mulMul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/mulП
lstm_3/lstm_cell_3/ReluRelu!lstm_3/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Reluі
lstm_3/lstm_cell_3/mul_1Mullstm_3/lstm_cell_3/Sigmoid:y:0%lstm_3/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/mul_1©
lstm_3/lstm_cell_3/add_1AddV2lstm_3/lstm_cell_3/mul:z:0lstm_3/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/add_1Ь
lstm_3/lstm_cell_3/Sigmoid_2Sigmoid!lstm_3/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Sigmoid_2О
lstm_3/lstm_cell_3/Relu_1Relulstm_3/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Relu_1Є
lstm_3/lstm_cell_3/mul_2Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0'lstm_3/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/mul_2Э
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2&
$lstm_3/TensorArrayV2_1/element_shape‘
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/timeН
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counterс
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_3_matmul_readvariableop_resource3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
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
lstm_3_while_body_154558*$
condR
lstm_3_while_cond_154557*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_3/while√
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStackП
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_3/strided_slice_3/stackК
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1К
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2ƒ
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_3/strided_slice_3З
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/permЅ
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimeb
lstm_4/ShapeShapelstm_3/transpose_1:y:0*
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
value	B :@2
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
value	B :@2
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
:€€€€€€€€€@2
lstm_4/zerosn
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:€€€€€€€€€@2
lstm_4/zeros_1Г
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose/permЯ
lstm_4/transpose	Transposelstm_3/transpose_1:y:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
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
valueB"€€€€    2>
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
lstm_4/lstm_cell_4/ones_likeК
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_4/lstm_cell_4/split/split_dimƒ
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_4/lstm_cell_4/split/ReadVariableOpу
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_4/lstm_cell_4/splitґ
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMulЇ
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_1Ї
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_2Ї
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_3О
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_4/lstm_cell_4/split_1/split_dim∆
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_4/lstm_cell_4/split_1/ReadVariableOpл
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_4/lstm_cell_4/split_1њ
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd≈
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd_1≈
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd_2≈
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd_3І
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mulЂ
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_1Ђ
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_2Ђ
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0%lstm_4/lstm_cell_4/ones_like:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_3≤
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2*
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

:@@*

begin_mask*
end_mask2"
 lstm_4/lstm_cell_4/strided_sliceљ
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_4Ј
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/addС
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Sigmoidґ
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_1•
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(lstm_4/lstm_cell_4/strided_slice_1/stack©
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2,
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

:@@*

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_1Ѕ
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_5љ
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_1Ч
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Sigmoid_1®
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_4ґ
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_2•
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2*
(lstm_4/lstm_cell_4/strided_slice_2/stack©
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2,
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

:@@*

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_2Ѕ
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_6љ
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_2К
lstm_4/lstm_cell_4/ReluRelulstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Reluі
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_5Ђ
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_3ґ
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_3•
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2*
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

:@@*

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_3Ѕ
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_7љ
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_4Ч
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Sigmoid_2О
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Relu_1Є
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_6Э
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2&
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_154748*$
condR
lstm_4_while_cond_154747*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
lstm_4/while√
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@*
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
:€€€€€€€€€@2
lstm_4/transpose_1t
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/runtime•
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_4/MatMul/ReadVariableOp§
dense_4/MatMulMatMullstm_4/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/MatMul§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp°
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/Relu•
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/BiasAddj
reshape_2/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/ShapeИ
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stackМ
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1М
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2Ю
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2“
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape£
reshape_2/ReshapeReshapedense_5/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_2/Reshapeў
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mulм
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muly
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityк
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp0^conv1d/kernel/Regularizer/Square/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*^lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_3/MatMul/ReadVariableOp+^lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^lstm_3/while"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2V
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp(lstm_3/lstm_cell_3/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while2F
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
:€€€€€€€€€
 
_user_specified_nameinputs
т
Г
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_151777

inputs

states
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
?:€€€€€€€€€@:€€€€€€€€€ :€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
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
E__inference_reshape_2_layer_call_and_return_conditional_losses_157362

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
–Q
Њ
B__inference_lstm_4_layer_call_and_return_conditional_losses_152696

inputs%
lstm_cell_4_152608:	 А!
lstm_cell_4_152610:	А%
lstm_cell_4_152612:	@А
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2	
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
 :€€€€€€€€€€€€€€€€€€ 2
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
valueB"€€€€    27
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
:€€€€€€€€€ *
shrink_axis_mask2
strided_slice_2Ч
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_152608lstm_cell_4_152610lstm_cell_4_152612*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1525432%
#lstm_cell_4/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_152608lstm_cell_4_152610lstm_cell_4_152612*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_152621*
condR
while_cond_152620*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   22
0TensorArrayV2Stack/TensorListStack/element_shapeс
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@*
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
:€€€€€€€€€@*
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
 :€€€€€€€€€€€€€€€€€€@2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeќ
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_cell_4_152608*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muls
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

IdentityЇ
NoOpNoOp<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ђ
∆
__inference_loss_fn_1_157716W
Dlstm_4_lstm_cell_4_kernel_regularizer_square_readvariableop_resource:	 А
identityИҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpА
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpDlstm_4_lstm_cell_4_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
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
’
√
while_cond_155617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_155617___redundant_placeholder04
0while_while_cond_155617___redundant_placeholder14
0while_while_cond_155617___redundant_placeholder24
0while_while_cond_155617___redundant_placeholder3
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
’
√
while_cond_155768
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_155768___redundant_placeholder04
0while_while_cond_155768___redundant_placeholder14
0while_while_cond_155768___redundant_placeholder24
0while_while_cond_155768___redundant_placeholder3
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
њ%
№
while_body_152621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_152645_0:	 А)
while_lstm_cell_4_152647_0:	А-
while_lstm_cell_4_152649_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_152645:	 А'
while_lstm_cell_4_152647:	А+
while_lstm_cell_4_152649:	@АИҐ)while/lstm_cell_4/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemџ
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_152645_0while_lstm_cell_4_152647_0while_lstm_cell_4_152649_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1525432+
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
:€€€€€€€€€@2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
while_lstm_cell_4_152645while_lstm_cell_4_152645_0"6
while_lstm_cell_4_152647while_lstm_cell_4_152647_0"6
while_lstm_cell_4_152649while_lstm_cell_4_152649_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2V
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
’
√
while_cond_153970
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_153970___redundant_placeholder04
0while_while_cond_153970___redundant_placeholder14
0while_while_cond_153970___redundant_placeholder24
0while_while_cond_153970___redundant_placeholder3
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
Р
e
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_155499

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
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

ExpandDims±
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolО
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€:e a
=
_output_shapes+
):'€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
њ%
№
while_body_152324
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_152348_0:	 А)
while_lstm_cell_4_152350_0:	А-
while_lstm_cell_4_152352_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_152348:	 А'
while_lstm_cell_4_152350:	А+
while_lstm_cell_4_152352:	@АИҐ)while/lstm_cell_4/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemџ
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_152348_0while_lstm_cell_4_152350_0while_lstm_cell_4_152352_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:€€€€€€€€€@:€€€€€€€€€@:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_1523102+
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
:€€€€€€€€€@2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
while_lstm_cell_4_152348while_lstm_cell_4_152348_0"6
while_lstm_cell_4_152350while_lstm_cell_4_152350_0"6
while_lstm_cell_4_152352while_lstm_cell_4_152352_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2V
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
’
√
while_cond_155919
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_155919___redundant_placeholder04
0while_while_cond_155919___redundant_placeholder14
0while_while_cond_155919___redundant_placeholder24
0while_while_cond_155919___redundant_placeholder3
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
Б
ф
C__inference_dense_4_layer_call_and_return_conditional_losses_153455

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
њ%
№
while_body_151855
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_3_151879_0:	@А-
while_lstm_cell_3_151881_0:	 А)
while_lstm_cell_3_151883_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_3_151879:	@А+
while_lstm_cell_3_151881:	 А'
while_lstm_cell_3_151883:	АИҐ)while/lstm_cell_3/StatefulPartitionedCall√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemџ
)while/lstm_cell_3/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_3_151879_0while_lstm_cell_3_151881_0while_lstm_cell_3_151883_0*
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1517772+
)while/lstm_cell_3/StatefulPartitionedCallц
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_3/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/lstm_cell_3/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_3/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5Ж

while/NoOpNoOp*^while/lstm_cell_3/StatefulPartitionedCall*"
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
while_lstm_cell_3_151879while_lstm_cell_3_151879_0"6
while_lstm_cell_3_151881while_lstm_cell_3_151881_0"6
while_lstm_cell_3_151883while_lstm_cell_3_151883_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2V
)while/lstm_cell_3/StatefulPartitionedCall)while/lstm_cell_3/StatefulPartitionedCall: 

_output_shapes
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
ъ
Е
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_157471

inputs
states_0
states_11
matmul_readvariableop_resource:	@А3
 matmul_1_readvariableop_resource:	 А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimњ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:€€€€€€€€€ 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2

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
?:€€€€€€€€€@:€€€€€€€€€ :€€€€€€€€€ : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
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
®

ѕ
lstm_3_while_cond_155003*
&lstm_3_while_lstm_3_while_loop_counter0
,lstm_3_while_lstm_3_while_maximum_iterations
lstm_3_while_placeholder
lstm_3_while_placeholder_1
lstm_3_while_placeholder_2
lstm_3_while_placeholder_3,
(lstm_3_while_less_lstm_3_strided_slice_1B
>lstm_3_while_lstm_3_while_cond_155003___redundant_placeholder0B
>lstm_3_while_lstm_3_while_cond_155003___redundant_placeholder1B
>lstm_3_while_lstm_3_while_cond_155003___redundant_placeholder2B
>lstm_3_while_lstm_3_while_cond_155003___redundant_placeholder3
lstm_3_while_identity
У
lstm_3/while/LessLesslstm_3_while_placeholder(lstm_3_while_less_lstm_3_strided_slice_1*
T0*
_output_shapes
: 2
lstm_3/while/Lessr
lstm_3/while/IdentityIdentitylstm_3/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_3/while/Identity"7
lstm_3_while_identitylstm_3/while/Identity:output:0*(
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
ц’
м
"__inference__traced_restore_158043
file_prefix4
assignvariableop_conv1d_kernel: ,
assignvariableop_1_conv1d_bias: 8
"assignvariableop_2_conv1d_1_kernel: @.
 assignvariableop_3_conv1d_1_bias:@3
!assignvariableop_4_dense_4_kernel:@@-
assignvariableop_5_dense_4_bias:@3
!assignvariableop_6_dense_5_kernel:@-
assignvariableop_7_dense_5_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: @
-assignvariableop_13_lstm_3_lstm_cell_3_kernel:	@АJ
7assignvariableop_14_lstm_3_lstm_cell_3_recurrent_kernel:	 А:
+assignvariableop_15_lstm_3_lstm_cell_3_bias:	А@
-assignvariableop_16_lstm_4_lstm_cell_4_kernel:	 АJ
7assignvariableop_17_lstm_4_lstm_cell_4_recurrent_kernel:	@А:
+assignvariableop_18_lstm_4_lstm_cell_4_bias:	А#
assignvariableop_19_total: #
assignvariableop_20_count: >
(assignvariableop_21_adam_conv1d_kernel_m: 4
&assignvariableop_22_adam_conv1d_bias_m: @
*assignvariableop_23_adam_conv1d_1_kernel_m: @6
(assignvariableop_24_adam_conv1d_1_bias_m:@;
)assignvariableop_25_adam_dense_4_kernel_m:@@5
'assignvariableop_26_adam_dense_4_bias_m:@;
)assignvariableop_27_adam_dense_5_kernel_m:@5
'assignvariableop_28_adam_dense_5_bias_m:G
4assignvariableop_29_adam_lstm_3_lstm_cell_3_kernel_m:	@АQ
>assignvariableop_30_adam_lstm_3_lstm_cell_3_recurrent_kernel_m:	 АA
2assignvariableop_31_adam_lstm_3_lstm_cell_3_bias_m:	АG
4assignvariableop_32_adam_lstm_4_lstm_cell_4_kernel_m:	 АQ
>assignvariableop_33_adam_lstm_4_lstm_cell_4_recurrent_kernel_m:	@АA
2assignvariableop_34_adam_lstm_4_lstm_cell_4_bias_m:	А>
(assignvariableop_35_adam_conv1d_kernel_v: 4
&assignvariableop_36_adam_conv1d_bias_v: @
*assignvariableop_37_adam_conv1d_1_kernel_v: @6
(assignvariableop_38_adam_conv1d_1_bias_v:@;
)assignvariableop_39_adam_dense_4_kernel_v:@@5
'assignvariableop_40_adam_dense_4_bias_v:@;
)assignvariableop_41_adam_dense_5_kernel_v:@5
'assignvariableop_42_adam_dense_5_bias_v:G
4assignvariableop_43_adam_lstm_3_lstm_cell_3_kernel_v:	@АQ
>assignvariableop_44_adam_lstm_3_lstm_cell_3_recurrent_kernel_v:	 АA
2assignvariableop_45_adam_lstm_3_lstm_cell_3_bias_v:	АG
4assignvariableop_46_adam_lstm_4_lstm_cell_4_kernel_v:	 АQ
>assignvariableop_47_adam_lstm_4_lstm_cell_4_recurrent_kernel_v:	@АA
2assignvariableop_48_adam_lstm_4_lstm_cell_4_bias_v:	А
identity_50ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9і
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ј
valueґB≥2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesт
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices®
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ё
_output_shapesЋ
»::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3•
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¶
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_4_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_4_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_5_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_5_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8°
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10І
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¶
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13µ
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_3_lstm_cell_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14њ
AssignVariableOp_14AssignVariableOp7assignvariableop_14_lstm_3_lstm_cell_3_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15≥
AssignVariableOp_15AssignVariableOp+assignvariableop_15_lstm_3_lstm_cell_3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16µ
AssignVariableOp_16AssignVariableOp-assignvariableop_16_lstm_4_lstm_cell_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17њ
AssignVariableOp_17AssignVariableOp7assignvariableop_17_lstm_4_lstm_cell_4_recurrent_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18≥
AssignVariableOp_18AssignVariableOp+assignvariableop_18_lstm_4_lstm_cell_4_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19°
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21∞
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_conv1d_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ѓ
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_conv1d_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≤
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv1d_1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24∞
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_4_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ѓ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_4_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27±
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_5_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ѓ
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_5_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Љ
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_3_lstm_cell_3_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30∆
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_lstm_3_lstm_cell_3_recurrent_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ї
AssignVariableOp_31AssignVariableOp2assignvariableop_31_adam_lstm_3_lstm_cell_3_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Љ
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_4_lstm_cell_4_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33∆
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_lstm_4_lstm_cell_4_recurrent_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Ї
AssignVariableOp_34AssignVariableOp2assignvariableop_34_adam_lstm_4_lstm_cell_4_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35∞
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv1d_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Ѓ
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv1d_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37≤
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38∞
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ѓ
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_5_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ѓ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_5_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Љ
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_lstm_3_lstm_cell_3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44∆
AssignVariableOp_44AssignVariableOp>assignvariableop_44_adam_lstm_3_lstm_cell_3_recurrent_kernel_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ї
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adam_lstm_3_lstm_cell_3_bias_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Љ
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_lstm_4_lstm_cell_4_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47∆
AssignVariableOp_47AssignVariableOp>assignvariableop_47_adam_lstm_4_lstm_cell_4_recurrent_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ї
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_lstm_4_lstm_cell_4_bias_vIdentity_48:output:0"/device:CPU:0*
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
Identity_50ь
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
ы
р
$__inference_signature_wrapper_154397
conv1d_input
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	@А
	unknown_4:	 А
	unknown_5:	А
	unknown_6:	 А
	unknown_7:	А
	unknown_8:	@А
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallconv1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_1515282
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv1d_input
Щ
±
__inference_loss_fn_0_157373N
8conv1d_kernel_regularizer_square_readvariableop_resource: 
identityИҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpя
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp8conv1d_kernel_regularizer_square_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mulk
IdentityIdentity!conv1d/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: 2

IdentityА
NoOpNoOp0^conv1d/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp
Щ
у
-__inference_sequential_1_layer_call_fn_154463

inputs
unknown: 
	unknown_0: 
	unknown_1: @
	unknown_2:@
	unknown_3:	@А
	unknown_4:	 А
	unknown_5:	А
	unknown_6:	 А
	unknown_7:	А
	unknown_8:	@А
	unknown_9:@@

unknown_10:@

unknown_11:@

unknown_12:
identityИҐStatefulPartitionedCallЫ
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
:€€€€€€€€€*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1541762
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
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
У[
Ф
B__inference_lstm_3_layer_call_and_return_conditional_losses_154055

inputs=
*lstm_cell_3_matmul_readvariableop_resource:	@А?
,lstm_cell_3_matmul_1_readvariableop_resource:	 А:
+lstm_cell_3_biasadd_readvariableop_resource:	А
identityИҐ"lstm_cell_3/BiasAdd/ReadVariableOpҐ!lstm_cell_3/MatMul/ReadVariableOpҐ#lstm_cell_3/MatMul_1/ReadVariableOpҐwhileD
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
:€€€€€€€€€@2
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
valueB"€€€€@   27
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
:€€€€€€€€€@*
shrink_axis_mask2
strided_slice_2≤
!lstm_cell_3/MatMul/ReadVariableOpReadVariableOp*lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02#
!lstm_cell_3/MatMul/ReadVariableOp™
lstm_cell_3/MatMulMatMulstrided_slice_2:output:0)lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMulЄ
#lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02%
#lstm_cell_3/MatMul_1/ReadVariableOp¶
lstm_cell_3/MatMul_1MatMulzeros:output:0+lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/MatMul_1Ь
lstm_cell_3/addAddV2lstm_cell_3/MatMul:product:0lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/add±
"lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02$
"lstm_cell_3/BiasAdd/ReadVariableOp©
lstm_cell_3/BiasAddBiasAddlstm_cell_3/add:z:0*lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_cell_3/BiasAdd|
lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_3/split/split_dimп
lstm_cell_3/splitSplit$lstm_cell_3/split/split_dim:output:0lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_cell_3/splitГ
lstm_cell_3/SigmoidSigmoidlstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/SigmoidЗ
lstm_cell_3/Sigmoid_1Sigmoidlstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_1И
lstm_cell_3/mulMullstm_cell_3/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mulz
lstm_cell_3/ReluRelulstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/ReluШ
lstm_cell_3/mul_1Mullstm_cell_3/Sigmoid:y:0lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_1Н
lstm_cell_3/add_1AddV2lstm_cell_3/mul:z:0lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/add_1З
lstm_cell_3/Sigmoid_2Sigmoidlstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Sigmoid_2y
lstm_cell_3/Relu_1Relulstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/Relu_1Ь
lstm_cell_3/mul_2Mullstm_cell_3/Sigmoid_2:y:0 lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_cell_3/mul_2П
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
while/loop_counterИ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_3_matmul_readvariableop_resource,lstm_cell_3_matmul_1_readvariableop_resource+lstm_cell_3_biasadd_readvariableop_resource*
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
while_body_153971*
condR
while_cond_153970*K
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
:€€€€€€€€€ *
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
:€€€€€€€€€ 2
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
:€€€€€€€€€ 2

Identity≈
NoOpNoOp#^lstm_cell_3/BiasAdd/ReadVariableOp"^lstm_cell_3/MatMul/ReadVariableOp$^lstm_cell_3/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€@: : : 2H
"lstm_cell_3/BiasAdd/ReadVariableOp"lstm_cell_3/BiasAdd/ReadVariableOp2F
!lstm_cell_3/MatMul/ReadVariableOp!lstm_cell_3/MatMul/ReadVariableOp2J
#lstm_cell_3/MatMul_1/ReadVariableOp#lstm_cell_3/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ы=
у
H__inference_sequential_1_layer_call_and_return_conditional_losses_154344
conv1d_input#
conv1d_154295: 
conv1d_154297: %
conv1d_1_154300: @
conv1d_1_154302:@ 
lstm_3_154306:	@А 
lstm_3_154308:	 А
lstm_3_154310:	А 
lstm_4_154313:	 А
lstm_4_154315:	А 
lstm_4_154317:	@А 
dense_4_154320:@@
dense_4_154322:@ 
dense_5_154325:@
dense_5_154327:
identityИҐconv1d/StatefulPartitionedCallҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpҐ conv1d_1/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐlstm_3/StatefulPartitionedCallҐlstm_4/StatefulPartitionedCallҐ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpФ
conv1d/StatefulPartitionedCallStatefulPartitionedCallconv1d_inputconv1d_154295conv1d_154297*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1529992 
conv1d/StatefulPartitionedCallє
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_154300conv1d_1_154302*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€
@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_conv1d_1_layer_call_and_return_conditional_losses_1530212"
 conv1d_1/StatefulPartitionedCallК
max_pooling1d/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_1530342
max_pooling1d/PartitionedCallњ
lstm_3/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0lstm_3_154306lstm_3_154308lstm_3_154310*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1540552 
lstm_3/StatefulPartitionedCallЉ
lstm_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_3/StatefulPartitionedCall:output:0lstm_4_154313lstm_4_154315lstm_4_154317*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1538822 
lstm_4/StatefulPartitionedCall∞
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_154320dense_4_154322*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1534552!
dense_4/StatefulPartitionedCall±
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_154325dense_5_154327*
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
C__inference_dense_5_layer_call_and_return_conditional_losses_1534712!
dense_5/StatefulPartitionedCallэ
reshape_2/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
E__inference_reshape_2_layer_call_and_return_conditional_losses_1534902
reshape_2/PartitionedCallі
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1d_154295*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mul…
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOplstm_4_154313*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/mulБ
IdentityIdentity"reshape_2/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

IdentityИ
NoOpNoOp^conv1d/StatefulPartitionedCall0^conv1d/kernel/Regularizer/Square/ReadVariableOp!^conv1d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall2z
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:Y U
+
_output_shapes
:€€€€€€€€€
&
_user_specified_nameconv1d_input
’
√
while_cond_156314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_156314___redundant_placeholder04
0while_while_cond_156314___redundant_placeholder14
0while_while_cond_156314___redundant_placeholder24
0while_while_cond_156314___redundant_placeholder3
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
@: : : : :€€€€€€€€€@:€€€€€€€€€@: ::::: 

_output_shapes
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
:
∆
F
*__inference_reshape_2_layer_call_fn_157349

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
E__inference_reshape_2_layer_call_and_return_conditional_losses_1534902
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
„
ґ
'__inference_lstm_3_layer_call_fn_155518
inputs_0
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_3_layer_call_and_return_conditional_losses_1517142
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@
"
_user_specified_name
inputs/0
¬>
«
while_body_156071
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_3_matmul_readvariableop_resource_0:	@АG
4while_lstm_cell_3_matmul_1_readvariableop_resource_0:	 АB
3while_lstm_cell_3_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_3_matmul_readvariableop_resource:	@АE
2while_lstm_cell_3_matmul_1_readvariableop_resource:	 А@
1while_lstm_cell_3_biasadd_readvariableop_resource:	АИҐ(while/lstm_cell_3/BiasAdd/ReadVariableOpҐ'while/lstm_cell_3/MatMul/ReadVariableOpҐ)while/lstm_cell_3/MatMul_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€@*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem∆
'while/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_3_matmul_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02)
'while/lstm_cell_3/MatMul/ReadVariableOp‘
while/lstm_cell_3/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMulћ
)while/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_3_matmul_1_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02+
)while/lstm_cell_3/MatMul_1/ReadVariableOpљ
while/lstm_cell_3/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/MatMul_1і
while/lstm_cell_3/addAddV2"while/lstm_cell_3/MatMul:product:0$while/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/add≈
(while/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_3_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_3/BiasAdd/ReadVariableOpЅ
while/lstm_cell_3/BiasAddBiasAddwhile/lstm_cell_3/add:z:00while/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
while/lstm_cell_3/BiasAddИ
!while/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_3/split/split_dimЗ
while/lstm_cell_3/splitSplit*while/lstm_cell_3/split/split_dim:output:0"while/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
while/lstm_cell_3/splitХ
while/lstm_cell_3/SigmoidSigmoid while/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/SigmoidЩ
while/lstm_cell_3/Sigmoid_1Sigmoid while/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_1Э
while/lstm_cell_3/mulMulwhile/lstm_cell_3/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mulМ
while/lstm_cell_3/ReluRelu while/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu∞
while/lstm_cell_3/mul_1Mulwhile/lstm_cell_3/Sigmoid:y:0$while/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_1•
while/lstm_cell_3/add_1AddV2while/lstm_cell_3/mul:z:0while/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/add_1Щ
while/lstm_cell_3/Sigmoid_2Sigmoid while/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Sigmoid_2Л
while/lstm_cell_3/Relu_1Reluwhile/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/Relu_1і
while/lstm_cell_3/mul_2Mulwhile/lstm_cell_3/Sigmoid_2:y:0&while/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/lstm_cell_3/mul_2я
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_3/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_3/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_3/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 2
while/Identity_5џ

while/NoOpNoOp)^while/lstm_cell_3/BiasAdd/ReadVariableOp(^while/lstm_cell_3/MatMul/ReadVariableOp*^while/lstm_cell_3/MatMul_1/ReadVariableOp*"
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
1while_lstm_cell_3_biasadd_readvariableop_resource3while_lstm_cell_3_biasadd_readvariableop_resource_0"j
2while_lstm_cell_3_matmul_1_readvariableop_resource4while_lstm_cell_3_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_3_matmul_readvariableop_resource2while_lstm_cell_3_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : 2T
(while/lstm_cell_3/BiasAdd/ReadVariableOp(while/lstm_cell_3/BiasAdd/ReadVariableOp2R
'while/lstm_cell_3/MatMul/ReadVariableOp'while/lstm_cell_3/MatMul/ReadVariableOp2V
)while/lstm_cell_3/MatMul_1/ReadVariableOp)while/lstm_cell_3/MatMul_1/ReadVariableOp: 

_output_shapes
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
Ыh
р
__inference__traced_save_157886
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableopB
>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop6
2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop8
4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop?
;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopI
Esavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop?
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
ShardedFilenameЃ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ј
valueґB≥2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesм
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЂ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_3_lstm_cell_3_kernel_read_readvariableop>savev2_lstm_3_lstm_cell_3_recurrent_kernel_read_readvariableop2savev2_lstm_3_lstm_cell_3_bias_read_readvariableop4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_m_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop;savev2_adam_lstm_3_lstm_cell_3_kernel_v_read_readvariableopEsavev2_adam_lstm_3_lstm_cell_3_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_3_lstm_cell_3_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422	2
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

identity_1Identity_1:output:0*≠
_input_shapesЫ
Ш: : : : @:@:@@:@:@:: : : : : :	@А:	 А:А:	 А:	@А:А: : : : : @:@:@@:@:@::	@А:	 А:А:	 А:	@А:А: : : @:@:@@:@:@::	@А:	 А:А:	 А:	@А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 
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

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 
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
: :%!

_output_shapes
:	@А:%!

_output_shapes
:	 А:!

_output_shapes	
:А:%!

_output_shapes
:	 А:%!

_output_shapes
:	@А:!

_output_shapes	
:А:

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
: : 
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

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::%!

_output_shapes
:	@А:%!

_output_shapes
:	 А:! 

_output_shapes	
:А:%!!

_output_shapes
:	 А:%"!

_output_shapes
:	@А:!#

_output_shapes	
:А:($$
"
_output_shapes
: : %
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

:@@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::%,!

_output_shapes
:	@А:%-!

_output_shapes
:	 А:!.

_output_shapes	
:А:%/!

_output_shapes
:	 А:%0!

_output_shapes
:	@А:!1

_output_shapes	
:А:2

_output_shapes
: 
Љ
ґ
'__inference_lstm_4_layer_call_fn_156183
inputs_0
unknown:	 А
	unknown_0:	А
	unknown_1:	@А
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_1526962
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
inputs/0
і
х
,__inference_lstm_cell_3_layer_call_fn_157390

inputs
states_0
states_1
unknown:	@А
	unknown_0:	 А
	unknown_1:	А
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_1516312
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
?:€€€€€€€€€@:€€€€€€€€€ :€€€€€€€€€ : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
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
Б
Ш
'__inference_conv1d_layer_call_fn_155434

inputs
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_1529992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Љ∞
Ш	
while_body_153717
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
1while_lstm_cell_4_split_readvariableop_resource_0:	 АB
3while_lstm_cell_4_split_1_readvariableop_resource_0:	А>
+while_lstm_cell_4_readvariableop_resource_0:	@А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
/while_lstm_cell_4_split_readvariableop_resource:	 А@
1while_lstm_cell_4_split_1_readvariableop_resource:	А<
)while_lstm_cell_4_readvariableop_resource:	@АИҐ while/lstm_cell_4/ReadVariableOpҐ"while/lstm_cell_4/ReadVariableOp_1Ґ"while/lstm_cell_4/ReadVariableOp_2Ґ"while/lstm_cell_4/ReadVariableOp_3Ґ&while/lstm_cell_4/split/ReadVariableOpҐ(while/lstm_cell_4/split_1/ReadVariableOp√
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape”
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/lstm_cell_4/dropout/MulЦ
while/lstm_cell_4/dropout/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2!
while/lstm_cell_4/dropout/ShapeЙ
6while/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform(while/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ІУК28
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
:€€€€€€€€€@2(
&while/lstm_cell_4/dropout/GreaterEqualµ
while/lstm_cell_4/dropout/CastCast*while/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2 
while/lstm_cell_4/dropout/Cast¬
while/lstm_cell_4/dropout/Mul_1Mul!while/lstm_cell_4/dropout/Mul:z:0"while/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2!
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_1/MulЪ
!while/lstm_cell_4/dropout_1/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_1/ShapeП
8while/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ВҐН2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_1/GreaterEqualї
 while/lstm_cell_4/dropout_1/CastCast,while/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_1/Cast 
!while/lstm_cell_4/dropout_1/Mul_1Mul#while/lstm_cell_4/dropout_1/Mul:z:0$while/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_2/MulЪ
!while/lstm_cell_4/dropout_2/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_2/ShapeП
8while/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2√тђ2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_2/GreaterEqualї
 while/lstm_cell_4/dropout_2/CastCast,while/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_2/Cast 
!while/lstm_cell_4/dropout_2/Mul_1Mul#while/lstm_cell_4/dropout_2/Mul:z:0$while/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
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
:€€€€€€€€€@2!
while/lstm_cell_4/dropout_3/MulЪ
!while/lstm_cell_4/dropout_3/ShapeShape$while/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2#
!while/lstm_cell_4/dropout_3/ShapeП
8while/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform*while/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2їЂъ2:
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
:€€€€€€€€€@2*
(while/lstm_cell_4/dropout_3/GreaterEqualї
 while/lstm_cell_4/dropout_3/CastCast,while/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2"
 while/lstm_cell_4/dropout_3/Cast 
!while/lstm_cell_4/dropout_3/Mul_1Mul#while/lstm_cell_4/dropout_3/Mul:z:0$while/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2#
!while/lstm_cell_4/dropout_3/Mul_1И
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim√
&while/lstm_cell_4/split/ReadVariableOpReadVariableOp1while_lstm_cell_4_split_readvariableop_resource_0*
_output_shapes
:	 А*
dtype02(
&while/lstm_cell_4/split/ReadVariableOpп
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0.while/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
while/lstm_cell_4/splitƒ
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul»
while/lstm_cell_4/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_1»
while/lstm_cell_4/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_2»
while/lstm_cell_4/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_3М
#while/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2%
#while/lstm_cell_4/split_1/split_dim≈
(while/lstm_cell_4/split_1/ReadVariableOpReadVariableOp3while_lstm_cell_4_split_1_readvariableop_resource_0*
_output_shapes	
:А*
dtype02*
(while/lstm_cell_4/split_1/ReadVariableOpз
while/lstm_cell_4/split_1Split,while/lstm_cell_4/split_1/split_dim:output:00while/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
while/lstm_cell_4/split_1ї
while/lstm_cell_4/BiasAddBiasAdd"while/lstm_cell_4/MatMul:product:0"while/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAddЅ
while/lstm_cell_4/BiasAdd_1BiasAdd$while/lstm_cell_4/MatMul_1:product:0"while/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_1Ѕ
while/lstm_cell_4/BiasAdd_2BiasAdd$while/lstm_cell_4/MatMul_2:product:0"while/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_2Ѕ
while/lstm_cell_4/BiasAdd_3BiasAdd$while/lstm_cell_4/MatMul_3:product:0"while/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/BiasAdd_3°
while/lstm_cell_4/mulMulwhile_placeholder_2#while/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mulІ
while/lstm_cell_4/mul_1Mulwhile_placeholder_2%while/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_1І
while/lstm_cell_4/mul_2Mulwhile_placeholder_2%while/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_2І
while/lstm_cell_4/mul_3Mulwhile_placeholder_2%while/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_3±
 while/lstm_cell_4/ReadVariableOpReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
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
valueB"    @   2)
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

:@@*

begin_mask*
end_mask2!
while/lstm_cell_4/strided_sliceє
while/lstm_cell_4/MatMul_4MatMulwhile/lstm_cell_4/mul:z:0(while/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_4≥
while/lstm_cell_4/addAddV2"while/lstm_cell_4/BiasAdd:output:0$while/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/addО
while/lstm_cell_4/SigmoidSigmoidwhile/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoidµ
"while/lstm_cell_4/ReadVariableOp_1ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_1£
'while/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2)
'while/lstm_cell_4/strided_slice_1/stackІ
)while/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_1љ
while/lstm_cell_4/MatMul_5MatMulwhile/lstm_cell_4/mul_1:z:0*while/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_5є
while/lstm_cell_4/add_1AddV2$while/lstm_cell_4/BiasAdd_1:output:0$while/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_1Ф
while/lstm_cell_4/Sigmoid_1Sigmoidwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_1°
while/lstm_cell_4/mul_4Mulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_4µ
"while/lstm_cell_4/ReadVariableOp_2ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_2£
'while/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2)
'while/lstm_cell_4/strided_slice_2/stackІ
)while/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2+
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_2љ
while/lstm_cell_4/MatMul_6MatMulwhile/lstm_cell_4/mul_2:z:0*while/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_6є
while/lstm_cell_4/add_2AddV2$while/lstm_cell_4/BiasAdd_2:output:0$while/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_2З
while/lstm_cell_4/ReluReluwhile/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu∞
while/lstm_cell_4/mul_5Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/mul_5І
while/lstm_cell_4/add_3AddV2while/lstm_cell_4/mul_4:z:0while/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_3µ
"while/lstm_cell_4/ReadVariableOp_3ReadVariableOp+while_lstm_cell_4_readvariableop_resource_0*
_output_shapes
:	@А*
dtype02$
"while/lstm_cell_4/ReadVariableOp_3£
'while/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2)
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

:@@*

begin_mask*
end_mask2#
!while/lstm_cell_4/strided_slice_3љ
while/lstm_cell_4/MatMul_7MatMulwhile/lstm_cell_4/mul_3:z:0*while/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/MatMul_7є
while/lstm_cell_4/add_4AddV2$while/lstm_cell_4/BiasAdd_3:output:0$while/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/add_4Ф
while/lstm_cell_4/Sigmoid_2Sigmoidwhile/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Sigmoid_2Л
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
while/lstm_cell_4/Relu_1і
while/lstm_cell_4/mul_6Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
:€€€€€€€€€@2
while/Identity_4М
while/Identity_5Identitywhile/lstm_cell_4/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:€€€€€€€€€@2
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : 2D
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
:€€€€€€€€€@:-)
'
_output_shapes
:€€€€€€€€€@:

_output_shapes
: :

_output_shapes
: 
УЫ
Б
H__inference_sequential_1_layer_call_and_return_conditional_losses_155419

inputsH
2conv1d_conv1d_expanddims_1_readvariableop_resource: 4
&conv1d_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource: @6
(conv1d_1_biasadd_readvariableop_resource:@D
1lstm_3_lstm_cell_3_matmul_readvariableop_resource:	@АF
3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource:	 АA
2lstm_3_lstm_cell_3_biasadd_readvariableop_resource:	АC
0lstm_4_lstm_cell_4_split_readvariableop_resource:	 АA
2lstm_4_lstm_cell_4_split_1_readvariableop_resource:	А=
*lstm_4_lstm_cell_4_readvariableop_resource:	@А8
&dense_4_matmul_readvariableop_resource:@@5
'dense_4_biasadd_readvariableop_resource:@8
&dense_5_matmul_readvariableop_resource:@5
'dense_5_biasadd_readvariableop_resource:
identityИҐconv1d/BiasAdd/ReadVariableOpҐ)conv1d/conv1d/ExpandDims_1/ReadVariableOpҐ/conv1d/kernel/Regularizer/Square/ReadVariableOpҐconv1d_1/BiasAdd/ReadVariableOpҐ+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐ)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpҐ(lstm_3/lstm_cell_3/MatMul/ReadVariableOpҐ*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpҐlstm_3/whileҐ!lstm_4/lstm_cell_4/ReadVariableOpҐ#lstm_4/lstm_cell_4/ReadVariableOp_1Ґ#lstm_4/lstm_cell_4/ReadVariableOp_2Ґ#lstm_4/lstm_cell_4/ReadVariableOp_3Ґ;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpҐ'lstm_4/lstm_cell_4/split/ReadVariableOpҐ)lstm_4/lstm_cell_4/split_1/ReadVariableOpҐlstm_4/whileЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2
conv1d/conv1d/ExpandDims/dimЂ
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv1d/conv1d/ExpandDimsЌ
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim”
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/conv1d/ExpandDims_1”
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv1d/conv1dІ
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€ *
squeeze_dims

э€€€€€€€€2
conv1d/conv1d/Squeeze°
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1d/BiasAdd/ReadVariableOp®
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
conv1d/BiasAddq
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
conv1d/ReluЛ
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
э€€€€€€€€2 
conv1d_1/conv1d/ExpandDims/dimƒ
conv1d_1/conv1d/ExpandDims
ExpandDimsconv1d/Relu:activations:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv1d_1/conv1d/ExpandDims”
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dimџ
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_1/conv1d/ExpandDims_1џ
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@*
paddingVALID*
strides
2
conv1d_1/conv1d≠
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@*
squeeze_dims

э€€€€€€€€2
conv1d_1/conv1d/SqueezeІ
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp∞
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
conv1d_1/BiasAddw
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€
@2
conv1d_1/Relu~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dimј
max_pooling1d/ExpandDims
ExpandDimsconv1d_1/Relu:activations:0%max_pooling1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:€€€€€€€€€
@2
max_pooling1d/ExpandDims…
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool¶
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*+
_output_shapes
:€€€€€€€€€@*
squeeze_dims
2
max_pooling1d/Squeezej
lstm_3/ShapeShapemax_pooling1d/Squeeze:output:0*
T0*
_output_shapes
:2
lstm_3/ShapeВ
lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice/stackЖ
lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_1Ж
lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_3/strided_slice/stack_2М
lstm_3/strided_sliceStridedSlicelstm_3/Shape:output:0#lstm_3/strided_slice/stack:output:0%lstm_3/strided_slice/stack_1:output:0%lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slicej
lstm_3/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/mul/yИ
lstm_3/zeros/mulMullstm_3/strided_slice:output:0lstm_3/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/mulm
lstm_3/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros/Less/yГ
lstm_3/zeros/LessLesslstm_3/zeros/mul:z:0lstm_3/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros/Lessp
lstm_3/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros/packed/1Я
lstm_3/zeros/packedPacklstm_3/strided_slice:output:0lstm_3/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros/packedm
lstm_3/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros/ConstС
lstm_3/zerosFilllstm_3/zeros/packed:output:0lstm_3/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/zerosn
lstm_3/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/mul/yО
lstm_3/zeros_1/mulMullstm_3/strided_slice:output:0lstm_3/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/mulq
lstm_3/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2
lstm_3/zeros_1/Less/yЛ
lstm_3/zeros_1/LessLesslstm_3/zeros_1/mul:z:0lstm_3/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_3/zeros_1/Lesst
lstm_3/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/zeros_1/packed/1•
lstm_3/zeros_1/packedPacklstm_3/strided_slice:output:0 lstm_3/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_3/zeros_1/packedq
lstm_3/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/zeros_1/ConstЩ
lstm_3/zeros_1Filllstm_3/zeros_1/packed:output:0lstm_3/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/zeros_1Г
lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose/permІ
lstm_3/transpose	Transposemax_pooling1d/Squeeze:output:0lstm_3/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€@2
lstm_3/transposed
lstm_3/Shape_1Shapelstm_3/transpose:y:0*
T0*
_output_shapes
:2
lstm_3/Shape_1Ж
lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_1/stackК
lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_1К
lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_1/stack_2Ш
lstm_3/strided_slice_1StridedSlicelstm_3/Shape_1:output:0%lstm_3/strided_slice_1/stack:output:0'lstm_3/strided_slice_1/stack_1:output:0'lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_3/strided_slice_1У
"lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2$
"lstm_3/TensorArrayV2/element_shapeќ
lstm_3/TensorArrayV2TensorListReserve+lstm_3/TensorArrayV2/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2Ќ
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2>
<lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeФ
.lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_3/transpose:y:0Elstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_3/TensorArrayUnstack/TensorListFromTensorЖ
lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_3/strided_slice_2/stackК
lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_1К
lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_2/stack_2¶
lstm_3/strided_slice_2StridedSlicelstm_3/transpose:y:0%lstm_3/strided_slice_2/stack:output:0'lstm_3/strided_slice_2/stack_1:output:0'lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€@*
shrink_axis_mask2
lstm_3/strided_slice_2«
(lstm_3/lstm_cell_3/MatMul/ReadVariableOpReadVariableOp1lstm_3_lstm_cell_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype02*
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp∆
lstm_3/lstm_cell_3/MatMulMatMullstm_3/strided_slice_2:output:00lstm_3/lstm_cell_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/MatMulЌ
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOpReadVariableOp3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource*
_output_shapes
:	 А*
dtype02,
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp¬
lstm_3/lstm_cell_3/MatMul_1MatMullstm_3/zeros:output:02lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/MatMul_1Є
lstm_3/lstm_cell_3/addAddV2#lstm_3/lstm_cell_3/MatMul:product:0%lstm_3/lstm_cell_3/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/add∆
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOpReadVariableOp2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp≈
lstm_3/lstm_cell_3/BiasAddBiasAddlstm_3/lstm_cell_3/add:z:01lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
lstm_3/lstm_cell_3/BiasAddК
"lstm_3/lstm_cell_3/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_3/lstm_cell_3/split/split_dimЛ
lstm_3/lstm_cell_3/splitSplit+lstm_3/lstm_cell_3/split/split_dim:output:0#lstm_3/lstm_cell_3/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split2
lstm_3/lstm_cell_3/splitШ
lstm_3/lstm_cell_3/SigmoidSigmoid!lstm_3/lstm_cell_3/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/SigmoidЬ
lstm_3/lstm_cell_3/Sigmoid_1Sigmoid!lstm_3/lstm_cell_3/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Sigmoid_1§
lstm_3/lstm_cell_3/mulMul lstm_3/lstm_cell_3/Sigmoid_1:y:0lstm_3/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/mulП
lstm_3/lstm_cell_3/ReluRelu!lstm_3/lstm_cell_3/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Reluі
lstm_3/lstm_cell_3/mul_1Mullstm_3/lstm_cell_3/Sigmoid:y:0%lstm_3/lstm_cell_3/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/mul_1©
lstm_3/lstm_cell_3/add_1AddV2lstm_3/lstm_cell_3/mul:z:0lstm_3/lstm_cell_3/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/add_1Ь
lstm_3/lstm_cell_3/Sigmoid_2Sigmoid!lstm_3/lstm_cell_3/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Sigmoid_2О
lstm_3/lstm_cell_3/Relu_1Relulstm_3/lstm_cell_3/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/Relu_1Є
lstm_3/lstm_cell_3/mul_2Mul lstm_3/lstm_cell_3/Sigmoid_2:y:0'lstm_3/lstm_cell_3/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€ 2
lstm_3/lstm_cell_3/mul_2Э
$lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    2&
$lstm_3/TensorArrayV2_1/element_shape‘
lstm_3/TensorArrayV2_1TensorListReserve-lstm_3/TensorArrayV2_1/element_shape:output:0lstm_3/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_3/TensorArrayV2_1\
lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/timeН
lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2!
lstm_3/while/maximum_iterationsx
lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_3/while/loop_counterс
lstm_3/whileWhile"lstm_3/while/loop_counter:output:0(lstm_3/while/maximum_iterations:output:0lstm_3/time:output:0lstm_3/TensorArrayV2_1:handle:0lstm_3/zeros:output:0lstm_3/zeros_1:output:0lstm_3/strided_slice_1:output:0>lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_3_lstm_cell_3_matmul_readvariableop_resource3lstm_3_lstm_cell_3_matmul_1_readvariableop_resource2lstm_3_lstm_cell_3_biasadd_readvariableop_resource*
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
lstm_3_while_body_155004*$
condR
lstm_3_while_cond_155003*K
output_shapes:
8: : : : :€€€€€€€€€ :€€€€€€€€€ : : : : : *
parallel_iterations 2
lstm_3/while√
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    29
7lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_3/TensorArrayV2Stack/TensorListStackTensorListStacklstm_3/while:output:3@lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€ *
element_dtype02+
)lstm_3/TensorArrayV2Stack/TensorListStackП
lstm_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€2
lstm_3/strided_slice_3/stackК
lstm_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_3/strided_slice_3/stack_1К
lstm_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_3/strided_slice_3/stack_2ƒ
lstm_3/strided_slice_3StridedSlice2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_3/strided_slice_3/stack:output:0'lstm_3/strided_slice_3/stack_1:output:0'lstm_3/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_mask2
lstm_3/strided_slice_3З
lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_3/transpose_1/permЅ
lstm_3/transpose_1	Transpose2lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_3/transpose_1/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
lstm_3/transpose_1t
lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_3/runtimeb
lstm_4/ShapeShapelstm_3/transpose_1:y:0*
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
value	B :@2
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
value	B :@2
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
:€€€€€€€€€@2
lstm_4/zerosn
lstm_4/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
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
value	B :@2
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
:€€€€€€€€€@2
lstm_4/zeros_1Г
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_4/transpose/permЯ
lstm_4/transpose	Transposelstm_3/transpose_1:y:0lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:€€€€€€€€€ 2
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
valueB"€€€€    2>
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
:€€€€€€€€€ *
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
:€€€€€€€€€@2
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
:€€€€€€€€€@2 
lstm_4/lstm_cell_4/dropout/MulЩ
 lstm_4/lstm_cell_4/dropout/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2"
 lstm_4/lstm_cell_4/dropout/ShapeМ
7lstm_4/lstm_cell_4/dropout/random_uniform/RandomUniformRandomUniform)lstm_4/lstm_cell_4/dropout/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ЎЬШ29
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
:€€€€€€€€€@2)
'lstm_4/lstm_cell_4/dropout/GreaterEqualЄ
lstm_4/lstm_cell_4/dropout/CastCast+lstm_4/lstm_cell_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2!
lstm_4/lstm_cell_4/dropout/Cast∆
 lstm_4/lstm_cell_4/dropout/Mul_1Mul"lstm_4/lstm_cell_4/dropout/Mul:z:0#lstm_4/lstm_cell_4/dropout/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2"
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
:€€€€€€€€€@2"
 lstm_4/lstm_cell_4/dropout_1/MulЭ
"lstm_4/lstm_cell_4/dropout_1/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/dropout_1/ShapeТ
9lstm_4/lstm_cell_4/dropout_1/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_1/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2Шбд2;
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
:€€€€€€€€€@2+
)lstm_4/lstm_cell_4/dropout_1/GreaterEqualЊ
!lstm_4/lstm_cell_4/dropout_1/CastCast-lstm_4/lstm_cell_4/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/lstm_cell_4/dropout_1/Castќ
"lstm_4/lstm_cell_4/dropout_1/Mul_1Mul$lstm_4/lstm_cell_4/dropout_1/Mul:z:0%lstm_4/lstm_cell_4/dropout_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
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
:€€€€€€€€€@2"
 lstm_4/lstm_cell_4/dropout_2/MulЭ
"lstm_4/lstm_cell_4/dropout_2/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/dropout_2/ShapeТ
9lstm_4/lstm_cell_4/dropout_2/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_2/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2З√І2;
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
:€€€€€€€€€@2+
)lstm_4/lstm_cell_4/dropout_2/GreaterEqualЊ
!lstm_4/lstm_cell_4/dropout_2/CastCast-lstm_4/lstm_cell_4/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/lstm_cell_4/dropout_2/Castќ
"lstm_4/lstm_cell_4/dropout_2/Mul_1Mul$lstm_4/lstm_cell_4/dropout_2/Mul:z:0%lstm_4/lstm_cell_4/dropout_2/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
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
:€€€€€€€€€@2"
 lstm_4/lstm_cell_4/dropout_3/MulЭ
"lstm_4/lstm_cell_4/dropout_3/ShapeShape%lstm_4/lstm_cell_4/ones_like:output:0*
T0*
_output_shapes
:2$
"lstm_4/lstm_cell_4/dropout_3/ShapeТ
9lstm_4/lstm_cell_4/dropout_3/random_uniform/RandomUniformRandomUniform+lstm_4/lstm_cell_4/dropout_3/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€@*
dtype0*
seed±€е)*
seed2ЄЂР2;
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
:€€€€€€€€€@2+
)lstm_4/lstm_cell_4/dropout_3/GreaterEqualЊ
!lstm_4/lstm_cell_4/dropout_3/CastCast-lstm_4/lstm_cell_4/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:€€€€€€€€€@2#
!lstm_4/lstm_cell_4/dropout_3/Castќ
"lstm_4/lstm_cell_4/dropout_3/Mul_1Mul$lstm_4/lstm_cell_4/dropout_3/Mul:z:0%lstm_4/lstm_cell_4/dropout_3/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€@2$
"lstm_4/lstm_cell_4/dropout_3/Mul_1К
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_4/lstm_cell_4/split/split_dimƒ
'lstm_4/lstm_cell_4/split/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02)
'lstm_4/lstm_cell_4/split/ReadVariableOpу
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0/lstm_4/lstm_cell_4/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(: @: @: @: @*
	num_split2
lstm_4/lstm_cell_4/splitґ
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMulЇ
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_1Ї
lstm_4/lstm_cell_4/MatMul_2MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_2Ї
lstm_4/lstm_cell_4/MatMul_3MatMullstm_4/strided_slice_2:output:0!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_3О
$lstm_4/lstm_cell_4/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$lstm_4/lstm_cell_4/split_1/split_dim∆
)lstm_4/lstm_cell_4/split_1/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_split_1_readvariableop_resource*
_output_shapes	
:А*
dtype02+
)lstm_4/lstm_cell_4/split_1/ReadVariableOpл
lstm_4/lstm_cell_4/split_1Split-lstm_4/lstm_cell_4/split_1/split_dim:output:01lstm_4/lstm_cell_4/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
:@:@:@:@*
	num_split2
lstm_4/lstm_cell_4/split_1њ
lstm_4/lstm_cell_4/BiasAddBiasAdd#lstm_4/lstm_cell_4/MatMul:product:0#lstm_4/lstm_cell_4/split_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd≈
lstm_4/lstm_cell_4/BiasAdd_1BiasAdd%lstm_4/lstm_cell_4/MatMul_1:product:0#lstm_4/lstm_cell_4/split_1:output:1*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd_1≈
lstm_4/lstm_cell_4/BiasAdd_2BiasAdd%lstm_4/lstm_cell_4/MatMul_2:product:0#lstm_4/lstm_cell_4/split_1:output:2*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd_2≈
lstm_4/lstm_cell_4/BiasAdd_3BiasAdd%lstm_4/lstm_cell_4/MatMul_3:product:0#lstm_4/lstm_cell_4/split_1:output:3*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/BiasAdd_3¶
lstm_4/lstm_cell_4/mulMullstm_4/zeros:output:0$lstm_4/lstm_cell_4/dropout/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mulђ
lstm_4/lstm_cell_4/mul_1Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_1ђ
lstm_4/lstm_cell_4/mul_2Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_2ђ
lstm_4/lstm_cell_4/mul_3Mullstm_4/zeros:output:0&lstm_4/lstm_cell_4/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_3≤
!lstm_4/lstm_cell_4/ReadVariableOpReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
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
valueB"    @   2*
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

:@@*

begin_mask*
end_mask2"
 lstm_4/lstm_cell_4/strided_sliceљ
lstm_4/lstm_cell_4/MatMul_4MatMullstm_4/lstm_cell_4/mul:z:0)lstm_4/lstm_cell_4/strided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_4Ј
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/BiasAdd:output:0%lstm_4/lstm_cell_4/MatMul_4:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/addС
lstm_4/lstm_cell_4/SigmoidSigmoidlstm_4/lstm_cell_4/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Sigmoidґ
#lstm_4/lstm_cell_4/ReadVariableOp_1ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_1•
(lstm_4/lstm_cell_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    @   2*
(lstm_4/lstm_cell_4/strided_slice_1/stack©
*lstm_4/lstm_cell_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    А   2,
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

:@@*

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_1Ѕ
lstm_4/lstm_cell_4/MatMul_5MatMullstm_4/lstm_cell_4/mul_1:z:0+lstm_4/lstm_cell_4/strided_slice_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_5љ
lstm_4/lstm_cell_4/add_1AddV2%lstm_4/lstm_cell_4/BiasAdd_1:output:0%lstm_4/lstm_cell_4/MatMul_5:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_1Ч
lstm_4/lstm_cell_4/Sigmoid_1Sigmoidlstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Sigmoid_1®
lstm_4/lstm_cell_4/mul_4Mul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_4ґ
#lstm_4/lstm_cell_4/ReadVariableOp_2ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_2•
(lstm_4/lstm_cell_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    А   2*
(lstm_4/lstm_cell_4/strided_slice_2/stack©
*lstm_4/lstm_cell_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ј   2,
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

:@@*

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_2Ѕ
lstm_4/lstm_cell_4/MatMul_6MatMullstm_4/lstm_cell_4/mul_2:z:0+lstm_4/lstm_cell_4/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_6љ
lstm_4/lstm_cell_4/add_2AddV2%lstm_4/lstm_cell_4/BiasAdd_2:output:0%lstm_4/lstm_cell_4/MatMul_6:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_2К
lstm_4/lstm_cell_4/ReluRelulstm_4/lstm_cell_4/add_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Reluі
lstm_4/lstm_cell_4/mul_5Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_5Ђ
lstm_4/lstm_cell_4/add_3AddV2lstm_4/lstm_cell_4/mul_4:z:0lstm_4/lstm_cell_4/mul_5:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_3ґ
#lstm_4/lstm_cell_4/ReadVariableOp_3ReadVariableOp*lstm_4_lstm_cell_4_readvariableop_resource*
_output_shapes
:	@А*
dtype02%
#lstm_4/lstm_cell_4/ReadVariableOp_3•
(lstm_4/lstm_cell_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    ј   2*
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

:@@*

begin_mask*
end_mask2$
"lstm_4/lstm_cell_4/strided_slice_3Ѕ
lstm_4/lstm_cell_4/MatMul_7MatMullstm_4/lstm_cell_4/mul_3:z:0+lstm_4/lstm_cell_4/strided_slice_3:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/MatMul_7љ
lstm_4/lstm_cell_4/add_4AddV2%lstm_4/lstm_cell_4/BiasAdd_3:output:0%lstm_4/lstm_cell_4/MatMul_7:product:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/add_4Ч
lstm_4/lstm_cell_4/Sigmoid_2Sigmoidlstm_4/lstm_cell_4/add_4:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Sigmoid_2О
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_3:z:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/Relu_1Є
lstm_4/lstm_cell_4/mul_6Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:€€€€€€€€€@2
lstm_4/lstm_cell_4/mul_6Э
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   2&
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
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_155226*$
condR
lstm_4_while_cond_155225*K
output_shapes:
8: : : : :€€€€€€€€€@:€€€€€€€€€@: : : : : *
parallel_iterations 2
lstm_4/while√
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€@   29
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeД
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:€€€€€€€€€@*
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
:€€€€€€€€€@*
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
:€€€€€€€€€@2
lstm_4/transpose_1t
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_4/runtime•
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
dense_4/MatMul/ReadVariableOp§
dense_4/MatMulMatMullstm_4/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/MatMul§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp°
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/Relu•
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_5/BiasAddj
reshape_2/ShapeShapedense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/ShapeИ
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stackМ
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1М
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2Ю
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2“
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape£
reshape_2/ReshapeReshapedense_5/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€2
reshape_2/Reshapeў
/conv1d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/conv1d/kernel/Regularizer/Square/ReadVariableOpі
 conv1d/kernel/Regularizer/SquareSquare7conv1d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*"
_output_shapes
: 2"
 conv1d/kernel/Regularizer/SquareЧ
conv1d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*!
valueB"          2!
conv1d/kernel/Regularizer/Constґ
conv1d/kernel/Regularizer/SumSum$conv1d/kernel/Regularizer/Square:y:0(conv1d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/SumЗ
conv1d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52!
conv1d/kernel/Regularizer/mul/xЄ
conv1d/kernel/Regularizer/mulMul(conv1d/kernel/Regularizer/mul/x:output:0&conv1d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1d/kernel/Regularizer/mulм
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0lstm_4_lstm_cell_4_split_readvariableop_resource*
_output_shapes
:	 А*
dtype02=
;lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp’
,lstm_4/lstm_cell_4/kernel/Regularizer/SquareSquareClstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	 А2.
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
 *љ7Ж52-
+lstm_4/lstm_cell_4/kernel/Regularizer/mul/xи
)lstm_4/lstm_cell_4/kernel/Regularizer/mulMul4lstm_4/lstm_cell_4/kernel/Regularizer/mul/x:output:02lstm_4/lstm_cell_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)lstm_4/lstm_cell_4/kernel/Regularizer/muly
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€2

Identityк
NoOpNoOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp0^conv1d/kernel/Regularizer/Square/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*^lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)^lstm_3/lstm_cell_3/MatMul/ReadVariableOp+^lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp^lstm_3/while"^lstm_4/lstm_cell_4/ReadVariableOp$^lstm_4/lstm_cell_4/ReadVariableOp_1$^lstm_4/lstm_cell_4/ReadVariableOp_2$^lstm_4/lstm_cell_4/ReadVariableOp_3<^lstm_4/lstm_cell_4/kernel/Regularizer/Square/ReadVariableOp(^lstm_4/lstm_cell_4/split/ReadVariableOp*^lstm_4/lstm_cell_4/split_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:€€€€€€€€€: : : : : : : : : : : : : : 2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2b
/conv1d/kernel/Regularizer/Square/ReadVariableOp/conv1d/kernel/Regularizer/Square/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2V
)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp)lstm_3/lstm_cell_3/BiasAdd/ReadVariableOp2T
(lstm_3/lstm_cell_3/MatMul/ReadVariableOp(lstm_3/lstm_cell_3/MatMul/ReadVariableOp2X
*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp*lstm_3/lstm_cell_3/MatMul_1/ReadVariableOp2
lstm_3/whilelstm_3/while2F
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
:€€€€€€€€€
 
_user_specified_nameinputs"®L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Њ
serving_default™
I
conv1d_input9
serving_default_conv1d_input:0€€€€€€€€€A
	reshape_24
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Нд
н
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
™__call__
Ђ_default_save_signature
+ђ&call_and_return_all_conditional_losses"
_tf_keras_sequential
љ

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"
_tf_keras_layer
І
regularization_losses
	variables
trainable_variables
	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"
_tf_keras_layer
≈
cell
 
state_spec
!regularization_losses
"trainable_variables
#	variables
$	keras_api
≥__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
≈
%cell
&
state_spec
'regularization_losses
(trainable_variables
)	variables
*	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
љ

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
І
7regularization_losses
8	variables
9trainable_variables
:	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
л
;iter

<beta_1

=beta_2
	>decay
?learning_ratemОmПmРmС+mТ,mУ1mФ2mХ@mЦAmЧBmШCmЩDmЪEmЫvЬvЭvЮvЯ+v†,v°1vҐ2v£@v§Av•Bv¶CvІDv®Ev©"
	optimizer
(
љ0"
trackable_list_wrapper
Ж
0
1
2
3
@4
A5
B6
C7
D8
E9
+10
,11
112
213"
trackable_list_wrapper
Ж
0
1
2
3
@4
A5
B6
C7
D8
E9
+10
,11
112
213"
trackable_list_wrapper
ќ
Flayer_regularization_losses

regularization_losses
Gmetrics
Hlayer_metrics
trainable_variables

Ilayers
	variables
Jnon_trainable_variables
™__call__
Ђ_default_save_signature
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
-
Њserving_default"
signature_map
#:! 2conv1d/kernel
: 2conv1d/bias
(
љ0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
Klayer_regularization_losses
regularization_losses
Lmetrics
	variables
Mlayer_metrics
trainable_variables

Nlayers
Onon_trainable_variables
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
%:# @2conv1d_1/kernel
:@2conv1d_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
Player_regularization_losses
regularization_losses
Qmetrics
	variables
Rlayer_metrics
trainable_variables

Slayers
Tnon_trainable_variables
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
Ulayer_regularization_losses
regularization_losses
Vmetrics
	variables
Wlayer_metrics
trainable_variables

Xlayers
Ynon_trainable_variables
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
г
Z
state_size

@kernel
Arecurrent_kernel
Bbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
Љ
_layer_regularization_losses
!regularization_losses
"trainable_variables
`metrics
alayer_metrics

bstates

clayers
#	variables
dnon_trainable_variables
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
г
e
state_size

Ckernel
Drecurrent_kernel
Ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
(
√0"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
Љ
jlayer_regularization_losses
'regularization_losses
(trainable_variables
kmetrics
llayer_metrics

mstates

nlayers
)	variables
onon_trainable_variables
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
 :@@2dense_4/kernel
:@2dense_4/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
∞
player_regularization_losses
-regularization_losses
qmetrics
.	variables
rlayer_metrics
/trainable_variables

slayers
tnon_trainable_variables
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
∞
ulayer_regularization_losses
3regularization_losses
vmetrics
4	variables
wlayer_metrics
5trainable_variables

xlayers
ynon_trainable_variables
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
zlayer_regularization_losses
7regularization_losses
{metrics
8	variables
|layer_metrics
9trainable_variables

}layers
~non_trainable_variables
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*	@А2lstm_3/lstm_cell_3/kernel
6:4	 А2#lstm_3/lstm_cell_3/recurrent_kernel
&:$А2lstm_3/lstm_cell_3/bias
,:*	 А2lstm_4/lstm_cell_4/kernel
6:4	@А2#lstm_4/lstm_cell_4/recurrent_kernel
&:$А2lstm_4/lstm_cell_4/bias
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
(
љ0"
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
5
@0
A1
B2"
trackable_list_wrapper
5
@0
A1
B2"
trackable_list_wrapper
µ
 Аlayer_regularization_losses
[regularization_losses
Бmetrics
\	variables
Вlayer_metrics
]trainable_variables
Гlayers
Дnon_trainable_variables
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
√0"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
µ
 Еlayer_regularization_losses
fregularization_losses
Жmetrics
g	variables
Зlayer_metrics
htrainable_variables
Иlayers
Йnon_trainable_variables
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
%0"
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
R

Кtotal

Лcount
М	variables
Н	keras_api"
_tf_keras_metric
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
(
√0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
К0
Л1"
trackable_list_wrapper
.
М	variables"
_generic_user_object
(:& 2Adam/conv1d/kernel/m
: 2Adam/conv1d/bias/m
*:( @2Adam/conv1d_1/kernel/m
 :@2Adam/conv1d_1/bias/m
%:#@@2Adam/dense_4/kernel/m
:@2Adam/dense_4/bias/m
%:#@2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
1:/	@А2 Adam/lstm_3/lstm_cell_3/kernel/m
;:9	 А2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/m
+:)А2Adam/lstm_3/lstm_cell_3/bias/m
1:/	 А2 Adam/lstm_4/lstm_cell_4/kernel/m
;:9	@А2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
+:)А2Adam/lstm_4/lstm_cell_4/bias/m
(:& 2Adam/conv1d/kernel/v
: 2Adam/conv1d/bias/v
*:( @2Adam/conv1d_1/kernel/v
 :@2Adam/conv1d_1/bias/v
%:#@@2Adam/dense_4/kernel/v
:@2Adam/dense_4/bias/v
%:#@2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
1:/	@А2 Adam/lstm_3/lstm_cell_3/kernel/v
;:9	 А2*Adam/lstm_3/lstm_cell_3/recurrent_kernel/v
+:)А2Adam/lstm_3/lstm_cell_3/bias/v
1:/	 А2 Adam/lstm_4/lstm_cell_4/kernel/v
;:9	@А2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
+:)А2Adam/lstm_4/lstm_cell_4/bias/v
В2€
-__inference_sequential_1_layer_call_fn_153536
-__inference_sequential_1_layer_call_fn_154430
-__inference_sequential_1_layer_call_fn_154463
-__inference_sequential_1_layer_call_fn_154240ј
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
—Bќ
!__inference__wrapped_model_151528conv1d_input"Ш
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
о2л
H__inference_sequential_1_layer_call_and_return_conditional_losses_154909
H__inference_sequential_1_layer_call_and_return_conditional_losses_155419
H__inference_sequential_1_layer_call_and_return_conditional_losses_154292
H__inference_sequential_1_layer_call_and_return_conditional_losses_154344ј
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
—2ќ
'__inference_conv1d_layer_call_fn_155434Ґ
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
B__inference_conv1d_layer_call_and_return_conditional_losses_155456Ґ
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
)__inference_conv1d_1_layer_call_fn_155465Ґ
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
D__inference_conv1d_1_layer_call_and_return_conditional_losses_155481Ґ
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
И2Е
.__inference_max_pooling1d_layer_call_fn_155486
.__inference_max_pooling1d_layer_call_fn_155491Ґ
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
Њ2ї
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_155499
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_155507Ґ
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
€2ь
'__inference_lstm_3_layer_call_fn_155518
'__inference_lstm_3_layer_call_fn_155529
'__inference_lstm_3_layer_call_fn_155540
'__inference_lstm_3_layer_call_fn_155551’
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
л2и
B__inference_lstm_3_layer_call_and_return_conditional_losses_155702
B__inference_lstm_3_layer_call_and_return_conditional_losses_155853
B__inference_lstm_3_layer_call_and_return_conditional_losses_156004
B__inference_lstm_3_layer_call_and_return_conditional_losses_156155’
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
'__inference_lstm_4_layer_call_fn_156172
'__inference_lstm_4_layer_call_fn_156183
'__inference_lstm_4_layer_call_fn_156194
'__inference_lstm_4_layer_call_fn_156205’
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
л2и
B__inference_lstm_4_layer_call_and_return_conditional_losses_156448
B__inference_lstm_4_layer_call_and_return_conditional_losses_156755
B__inference_lstm_4_layer_call_and_return_conditional_losses_156998
B__inference_lstm_4_layer_call_and_return_conditional_losses_157305’
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
“2ѕ
(__inference_dense_4_layer_call_fn_157314Ґ
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
C__inference_dense_4_layer_call_and_return_conditional_losses_157325Ґ
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
(__inference_dense_5_layer_call_fn_157334Ґ
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
C__inference_dense_5_layer_call_and_return_conditional_losses_157344Ґ
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
*__inference_reshape_2_layer_call_fn_157349Ґ
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
E__inference_reshape_2_layer_call_and_return_conditional_losses_157362Ґ
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
__inference_loss_fn_0_157373П
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
–BЌ
$__inference_signature_wrapper_154397conv1d_input"Ф
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
†2Э
,__inference_lstm_cell_3_layer_call_fn_157390
,__inference_lstm_cell_3_layer_call_fn_157407Њ
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
÷2”
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_157439
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_157471Њ
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
,__inference_lstm_cell_4_layer_call_fn_157494
,__inference_lstm_cell_4_layer_call_fn_157511Њ
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
÷2”
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_157592
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_157705Њ
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
__inference_loss_fn_1_157716П
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
annotations™ *Ґ ђ
!__inference__wrapped_model_151528Ж@ABCED+,129Ґ6
/Ґ,
*К'
conv1d_input€€€€€€€€€
™ "9™6
4
	reshape_2'К$
	reshape_2€€€€€€€€€ђ
D__inference_conv1d_1_layer_call_and_return_conditional_losses_155481d3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€ 
™ ")Ґ&
К
0€€€€€€€€€
@
Ъ Д
)__inference_conv1d_1_layer_call_fn_155465W3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€ 
™ "К€€€€€€€€€
@™
B__inference_conv1d_layer_call_and_return_conditional_losses_155456d3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ В
'__inference_conv1d_layer_call_fn_155434W3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "К€€€€€€€€€ £
C__inference_dense_4_layer_call_and_return_conditional_losses_157325\+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€@
Ъ {
(__inference_dense_4_layer_call_fn_157314O+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€@£
C__inference_dense_5_layer_call_and_return_conditional_losses_157344\12/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ {
(__inference_dense_5_layer_call_fn_157334O12/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€;
__inference_loss_fn_0_157373Ґ

Ґ 
™ "К ;
__inference_loss_fn_1_157716CҐ

Ґ 
™ "К —
B__inference_lstm_3_layer_call_and_return_conditional_losses_155702К@ABOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p 

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ —
B__inference_lstm_3_layer_call_and_return_conditional_losses_155853К@ABOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p

 
™ "2Ґ/
(К%
0€€€€€€€€€€€€€€€€€€ 
Ъ Ј
B__inference_lstm_3_layer_call_and_return_conditional_losses_156004q@AB?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€@

 
p 

 
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ Ј
B__inference_lstm_3_layer_call_and_return_conditional_losses_156155q@AB?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€@

 
p

 
™ ")Ґ&
К
0€€€€€€€€€ 
Ъ ®
'__inference_lstm_3_layer_call_fn_155518}@ABOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p 

 
™ "%К"€€€€€€€€€€€€€€€€€€ ®
'__inference_lstm_3_layer_call_fn_155529}@ABOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€@

 
p

 
™ "%К"€€€€€€€€€€€€€€€€€€ П
'__inference_lstm_3_layer_call_fn_155540d@AB?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€@

 
p 

 
™ "К€€€€€€€€€ П
'__inference_lstm_3_layer_call_fn_155551d@AB?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€@

 
p

 
™ "К€€€€€€€€€ √
B__inference_lstm_4_layer_call_and_return_conditional_losses_156448}CEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ √
B__inference_lstm_4_layer_call_and_return_conditional_losses_156755}CEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≥
B__inference_lstm_4_layer_call_and_return_conditional_losses_156998mCED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p 

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ ≥
B__inference_lstm_4_layer_call_and_return_conditional_losses_157305mCED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p

 
™ "%Ґ"
К
0€€€€€€€€€@
Ъ Ы
'__inference_lstm_4_layer_call_fn_156172pCEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p 

 
™ "К€€€€€€€€€@Ы
'__inference_lstm_4_layer_call_fn_156183pCEDOҐL
EҐB
4Ъ1
/К,
inputs/0€€€€€€€€€€€€€€€€€€ 

 
p

 
™ "К€€€€€€€€€@Л
'__inference_lstm_4_layer_call_fn_156194`CED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p 

 
™ "К€€€€€€€€€@Л
'__inference_lstm_4_layer_call_fn_156205`CED?Ґ<
5Ґ2
$К!
inputs€€€€€€€€€ 

 
p

 
™ "К€€€€€€€€€@…
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_157439э@ABАҐ}
vҐs
 К
inputs€€€€€€€€€@
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
G__inference_lstm_cell_3_layer_call_and_return_conditional_losses_157471э@ABАҐ}
vҐs
 К
inputs€€€€€€€€€@
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
,__inference_lstm_cell_3_layer_call_fn_157390н@ABАҐ}
vҐs
 К
inputs€€€€€€€€€@
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
,__inference_lstm_cell_3_layer_call_fn_157407н@ABАҐ}
vҐs
 К
inputs€€€€€€€€€@
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
1/1€€€€€€€€€ …
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_157592эCEDАҐ}
vҐs
 К
inputs€€€€€€€€€ 
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ …
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_157705эCEDАҐ}
vҐs
 К
inputs€€€€€€€€€ 
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "sҐp
iҐf
К
0/0€€€€€€€€€@
EЪB
К
0/1/0€€€€€€€€€@
К
0/1/1€€€€€€€€€@
Ъ Ю
,__inference_lstm_cell_4_layer_call_fn_157494нCEDАҐ}
vҐs
 К
inputs€€€€€€€€€ 
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p 
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@Ю
,__inference_lstm_cell_4_layer_call_fn_157511нCEDАҐ}
vҐs
 К
inputs€€€€€€€€€ 
KҐH
"К
states/0€€€€€€€€€@
"К
states/1€€€€€€€€€@
p
™ "cҐ`
К
0€€€€€€€€€@
AЪ>
К
1/0€€€€€€€€€@
К
1/1€€€€€€€€€@“
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_155499ДEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";Ґ8
1К.
0'€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≠
I__inference_max_pooling1d_layer_call_and_return_conditional_losses_155507`3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
@
™ ")Ґ&
К
0€€€€€€€€€@
Ъ ©
.__inference_max_pooling1d_layer_call_fn_155486wEҐB
;Ґ8
6К3
inputs'€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ".К+'€€€€€€€€€€€€€€€€€€€€€€€€€€€Е
.__inference_max_pooling1d_layer_call_fn_155491S3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
@
™ "К€€€€€€€€€@•
E__inference_reshape_2_layer_call_and_return_conditional_losses_157362\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ")Ґ&
К
0€€€€€€€€€
Ъ }
*__inference_reshape_2_layer_call_fn_157349O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€ 
H__inference_sequential_1_layer_call_and_return_conditional_losses_154292~@ABCED+,12AҐ>
7Ґ4
*К'
conv1d_input€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ  
H__inference_sequential_1_layer_call_and_return_conditional_losses_154344~@ABCED+,12AҐ>
7Ґ4
*К'
conv1d_input€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ƒ
H__inference_sequential_1_layer_call_and_return_conditional_losses_154909x@ABCED+,12;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ ƒ
H__inference_sequential_1_layer_call_and_return_conditional_losses_155419x@ABCED+,12;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ ")Ґ&
К
0€€€€€€€€€
Ъ Ґ
-__inference_sequential_1_layer_call_fn_153536q@ABCED+,12AҐ>
7Ґ4
*К'
conv1d_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ґ
-__inference_sequential_1_layer_call_fn_154240q@ABCED+,12AҐ>
7Ґ4
*К'
conv1d_input€€€€€€€€€
p

 
™ "К€€€€€€€€€Ь
-__inference_sequential_1_layer_call_fn_154430k@ABCED+,12;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€Ь
-__inference_sequential_1_layer_call_fn_154463k@ABCED+,12;Ґ8
1Ґ.
$К!
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€њ
$__inference_signature_wrapper_154397Ц@ABCED+,12IҐF
Ґ 
?™<
:
conv1d_input*К'
conv1d_input€€€€€€€€€"9™6
4
	reshape_2'К$
	reshape_2€€€€€€€€€