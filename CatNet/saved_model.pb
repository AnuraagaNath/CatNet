��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense_1/bias/*
dtype0*
shape:*$
shared_nameAdam/v/dense_1/bias
w
'Adam/v/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/bias*
_output_shapes
:*
dtype0
�
Adam/m/dense_1/biasVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense_1/bias/*
dtype0*
shape:*$
shared_nameAdam/m/dense_1/bias
w
'Adam/m/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/bias*
_output_shapes
:*
dtype0
�
Adam/v/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/dense_1/kernel/*
dtype0*
shape:	�*&
shared_nameAdam/v/dense_1/kernel
�
)Adam/v/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/m/dense_1/kernelVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/dense_1/kernel/*
dtype0*
shape:	�*&
shared_nameAdam/m/dense_1/kernel
�
)Adam/m/dense_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/v/dense/bias/*
dtype0*
shape:�*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *"

debug_nameAdam/m/dense/bias/*
dtype0*
shape:�*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/v/dense/kernel/*
dtype0*
shape:���*$
shared_nameAdam/v/dense/kernel
~
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*!
_output_shapes
:���*
dtype0
�
Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *$

debug_nameAdam/m/dense/kernel/*
dtype0*
shape:���*$
shared_nameAdam/m/dense/kernel
~
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*!
_output_shapes
:���*
dtype0
�
-Adam/v/cat_block_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *>

debug_name0.Adam/v/cat_block_2/batch_normalization_2/beta/*
dtype0*
shape:�*>
shared_name/-Adam/v/cat_block_2/batch_normalization_2/beta
�
AAdam/v/cat_block_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp-Adam/v/cat_block_2/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
-Adam/m/cat_block_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *>

debug_name0.Adam/m/cat_block_2/batch_normalization_2/beta/*
dtype0*
shape:�*>
shared_name/-Adam/m/cat_block_2/batch_normalization_2/beta
�
AAdam/m/cat_block_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp-Adam/m/cat_block_2/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
.Adam/v/cat_block_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *?

debug_name1/Adam/v/cat_block_2/batch_normalization_2/gamma/*
dtype0*
shape:�*?
shared_name0.Adam/v/cat_block_2/batch_normalization_2/gamma
�
BAdam/v/cat_block_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp.Adam/v/cat_block_2/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
.Adam/m/cat_block_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *?

debug_name1/Adam/m/cat_block_2/batch_normalization_2/gamma/*
dtype0*
shape:�*?
shared_name0.Adam/m/cat_block_2/batch_normalization_2/gamma
�
BAdam/m/cat_block_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp.Adam/m/cat_block_2/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
 Adam/v/cat_block_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *1

debug_name#!Adam/v/cat_block_2/conv2d_2/bias/*
dtype0*
shape:�*1
shared_name" Adam/v/cat_block_2/conv2d_2/bias
�
4Adam/v/cat_block_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOp Adam/v/cat_block_2/conv2d_2/bias*
_output_shapes	
:�*
dtype0
�
 Adam/m/cat_block_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *1

debug_name#!Adam/m/cat_block_2/conv2d_2/bias/*
dtype0*
shape:�*1
shared_name" Adam/m/cat_block_2/conv2d_2/bias
�
4Adam/m/cat_block_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOp Adam/m/cat_block_2/conv2d_2/bias*
_output_shapes	
:�*
dtype0
�
"Adam/v/cat_block_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/cat_block_2/conv2d_2/kernel/*
dtype0*
shape:��*3
shared_name$"Adam/v/cat_block_2/conv2d_2/kernel
�
6Adam/v/cat_block_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/cat_block_2/conv2d_2/kernel*(
_output_shapes
:��*
dtype0
�
"Adam/m/cat_block_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/cat_block_2/conv2d_2/kernel/*
dtype0*
shape:��*3
shared_name$"Adam/m/cat_block_2/conv2d_2/kernel
�
6Adam/m/cat_block_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/cat_block_2/conv2d_2/kernel*(
_output_shapes
:��*
dtype0
�
-Adam/v/cat_block_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *>

debug_name0.Adam/v/cat_block_1/batch_normalization_1/beta/*
dtype0*
shape:�*>
shared_name/-Adam/v/cat_block_1/batch_normalization_1/beta
�
AAdam/v/cat_block_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp-Adam/v/cat_block_1/batch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
-Adam/m/cat_block_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *>

debug_name0.Adam/m/cat_block_1/batch_normalization_1/beta/*
dtype0*
shape:�*>
shared_name/-Adam/m/cat_block_1/batch_normalization_1/beta
�
AAdam/m/cat_block_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp-Adam/m/cat_block_1/batch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
.Adam/v/cat_block_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *?

debug_name1/Adam/v/cat_block_1/batch_normalization_1/gamma/*
dtype0*
shape:�*?
shared_name0.Adam/v/cat_block_1/batch_normalization_1/gamma
�
BAdam/v/cat_block_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp.Adam/v/cat_block_1/batch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
.Adam/m/cat_block_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *?

debug_name1/Adam/m/cat_block_1/batch_normalization_1/gamma/*
dtype0*
shape:�*?
shared_name0.Adam/m/cat_block_1/batch_normalization_1/gamma
�
BAdam/m/cat_block_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp.Adam/m/cat_block_1/batch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
 Adam/v/cat_block_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *1

debug_name#!Adam/v/cat_block_1/conv2d_1/bias/*
dtype0*
shape:�*1
shared_name" Adam/v/cat_block_1/conv2d_1/bias
�
4Adam/v/cat_block_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOp Adam/v/cat_block_1/conv2d_1/bias*
_output_shapes	
:�*
dtype0
�
 Adam/m/cat_block_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *1

debug_name#!Adam/m/cat_block_1/conv2d_1/bias/*
dtype0*
shape:�*1
shared_name" Adam/m/cat_block_1/conv2d_1/bias
�
4Adam/m/cat_block_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOp Adam/m/cat_block_1/conv2d_1/bias*
_output_shapes	
:�*
dtype0
�
"Adam/v/cat_block_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/cat_block_1/conv2d_1/kernel/*
dtype0*
shape:@�*3
shared_name$"Adam/v/cat_block_1/conv2d_1/kernel
�
6Adam/v/cat_block_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp"Adam/v/cat_block_1/conv2d_1/kernel*'
_output_shapes
:@�*
dtype0
�
"Adam/m/cat_block_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/cat_block_1/conv2d_1/kernel/*
dtype0*
shape:@�*3
shared_name$"Adam/m/cat_block_1/conv2d_1/kernel
�
6Adam/m/cat_block_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp"Adam/m/cat_block_1/conv2d_1/kernel*'
_output_shapes
:@�*
dtype0
�
)Adam/v/cat_block/batch_normalization/betaVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/v/cat_block/batch_normalization/beta/*
dtype0*
shape:@*:
shared_name+)Adam/v/cat_block/batch_normalization/beta
�
=Adam/v/cat_block/batch_normalization/beta/Read/ReadVariableOpReadVariableOp)Adam/v/cat_block/batch_normalization/beta*
_output_shapes
:@*
dtype0
�
)Adam/m/cat_block/batch_normalization/betaVarHandleOp*
_output_shapes
: *:

debug_name,*Adam/m/cat_block/batch_normalization/beta/*
dtype0*
shape:@*:
shared_name+)Adam/m/cat_block/batch_normalization/beta
�
=Adam/m/cat_block/batch_normalization/beta/Read/ReadVariableOpReadVariableOp)Adam/m/cat_block/batch_normalization/beta*
_output_shapes
:@*
dtype0
�
*Adam/v/cat_block/batch_normalization/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+Adam/v/cat_block/batch_normalization/gamma/*
dtype0*
shape:@*;
shared_name,*Adam/v/cat_block/batch_normalization/gamma
�
>Adam/v/cat_block/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp*Adam/v/cat_block/batch_normalization/gamma*
_output_shapes
:@*
dtype0
�
*Adam/m/cat_block/batch_normalization/gammaVarHandleOp*
_output_shapes
: *;

debug_name-+Adam/m/cat_block/batch_normalization/gamma/*
dtype0*
shape:@*;
shared_name,*Adam/m/cat_block/batch_normalization/gamma
�
>Adam/m/cat_block/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp*Adam/m/cat_block/batch_normalization/gamma*
_output_shapes
:@*
dtype0
�
Adam/v/cat_block/conv2d/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/cat_block/conv2d/bias/*
dtype0*
shape:@*-
shared_nameAdam/v/cat_block/conv2d/bias
�
0Adam/v/cat_block/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/v/cat_block/conv2d/bias*
_output_shapes
:@*
dtype0
�
Adam/m/cat_block/conv2d/biasVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/cat_block/conv2d/bias/*
dtype0*
shape:@*-
shared_nameAdam/m/cat_block/conv2d/bias
�
0Adam/m/cat_block/conv2d/bias/Read/ReadVariableOpReadVariableOpAdam/m/cat_block/conv2d/bias*
_output_shapes
:@*
dtype0
�
Adam/v/cat_block/conv2d/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/cat_block/conv2d/kernel/*
dtype0*
shape:@*/
shared_name Adam/v/cat_block/conv2d/kernel
�
2Adam/v/cat_block/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/v/cat_block/conv2d/kernel*&
_output_shapes
:@*
dtype0
�
Adam/m/cat_block/conv2d/kernelVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/cat_block/conv2d/kernel/*
dtype0*
shape:@*/
shared_name Adam/m/cat_block/conv2d/kernel
�
2Adam/m/cat_block/conv2d/kernel/Read/ReadVariableOpReadVariableOpAdam/m/cat_block/conv2d/kernel*&
_output_shapes
:@*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
�
dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:�*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:�*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape:���*
shared_namedense/kernel
p
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*!
_output_shapes
:���*
dtype0
�
1cat_block_2/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *B

debug_name42cat_block_2/batch_normalization_2/moving_variance/*
dtype0*
shape:�*B
shared_name31cat_block_2/batch_normalization_2/moving_variance
�
Ecat_block_2/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp1cat_block_2/batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
-cat_block_2/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *>

debug_name0.cat_block_2/batch_normalization_2/moving_mean/*
dtype0*
shape:�*>
shared_name/-cat_block_2/batch_normalization_2/moving_mean
�
Acat_block_2/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp-cat_block_2/batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
&cat_block_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *7

debug_name)'cat_block_2/batch_normalization_2/beta/*
dtype0*
shape:�*7
shared_name(&cat_block_2/batch_normalization_2/beta
�
:cat_block_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp&cat_block_2/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
'cat_block_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *8

debug_name*(cat_block_2/batch_normalization_2/gamma/*
dtype0*
shape:�*8
shared_name)'cat_block_2/batch_normalization_2/gamma
�
;cat_block_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp'cat_block_2/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
cat_block_2/conv2d_2/biasVarHandleOp*
_output_shapes
: **

debug_namecat_block_2/conv2d_2/bias/*
dtype0*
shape:�**
shared_namecat_block_2/conv2d_2/bias
�
-cat_block_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOpcat_block_2/conv2d_2/bias*
_output_shapes	
:�*
dtype0
�
cat_block_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *,

debug_namecat_block_2/conv2d_2/kernel/*
dtype0*
shape:��*,
shared_namecat_block_2/conv2d_2/kernel
�
/cat_block_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpcat_block_2/conv2d_2/kernel*(
_output_shapes
:��*
dtype0
�
1cat_block_1/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *B

debug_name42cat_block_1/batch_normalization_1/moving_variance/*
dtype0*
shape:�*B
shared_name31cat_block_1/batch_normalization_1/moving_variance
�
Ecat_block_1/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp1cat_block_1/batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
-cat_block_1/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *>

debug_name0.cat_block_1/batch_normalization_1/moving_mean/*
dtype0*
shape:�*>
shared_name/-cat_block_1/batch_normalization_1/moving_mean
�
Acat_block_1/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp-cat_block_1/batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
&cat_block_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *7

debug_name)'cat_block_1/batch_normalization_1/beta/*
dtype0*
shape:�*7
shared_name(&cat_block_1/batch_normalization_1/beta
�
:cat_block_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp&cat_block_1/batch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
'cat_block_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *8

debug_name*(cat_block_1/batch_normalization_1/gamma/*
dtype0*
shape:�*8
shared_name)'cat_block_1/batch_normalization_1/gamma
�
;cat_block_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp'cat_block_1/batch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
cat_block_1/conv2d_1/biasVarHandleOp*
_output_shapes
: **

debug_namecat_block_1/conv2d_1/bias/*
dtype0*
shape:�**
shared_namecat_block_1/conv2d_1/bias
�
-cat_block_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpcat_block_1/conv2d_1/bias*
_output_shapes	
:�*
dtype0
�
cat_block_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *,

debug_namecat_block_1/conv2d_1/kernel/*
dtype0*
shape:@�*,
shared_namecat_block_1/conv2d_1/kernel
�
/cat_block_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpcat_block_1/conv2d_1/kernel*'
_output_shapes
:@�*
dtype0
�
-cat_block/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *>

debug_name0.cat_block/batch_normalization/moving_variance/*
dtype0*
shape:@*>
shared_name/-cat_block/batch_normalization/moving_variance
�
Acat_block/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp-cat_block/batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
�
)cat_block/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *:

debug_name,*cat_block/batch_normalization/moving_mean/*
dtype0*
shape:@*:
shared_name+)cat_block/batch_normalization/moving_mean
�
=cat_block/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp)cat_block/batch_normalization/moving_mean*
_output_shapes
:@*
dtype0
�
"cat_block/batch_normalization/betaVarHandleOp*
_output_shapes
: *3

debug_name%#cat_block/batch_normalization/beta/*
dtype0*
shape:@*3
shared_name$"cat_block/batch_normalization/beta
�
6cat_block/batch_normalization/beta/Read/ReadVariableOpReadVariableOp"cat_block/batch_normalization/beta*
_output_shapes
:@*
dtype0
�
#cat_block/batch_normalization/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$cat_block/batch_normalization/gamma/*
dtype0*
shape:@*4
shared_name%#cat_block/batch_normalization/gamma
�
7cat_block/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp#cat_block/batch_normalization/gamma*
_output_shapes
:@*
dtype0
�
cat_block/conv2d/biasVarHandleOp*
_output_shapes
: *&

debug_namecat_block/conv2d/bias/*
dtype0*
shape:@*&
shared_namecat_block/conv2d/bias
{
)cat_block/conv2d/bias/Read/ReadVariableOpReadVariableOpcat_block/conv2d/bias*
_output_shapes
:@*
dtype0
�
cat_block/conv2d/kernelVarHandleOp*
_output_shapes
: *(

debug_namecat_block/conv2d/kernel/*
dtype0*
shape:@*(
shared_namecat_block/conv2d/kernel
�
+cat_block/conv2d/kernel/Read/ReadVariableOpReadVariableOpcat_block/conv2d/kernel*&
_output_shapes
:@*
dtype0
�
serving_default_input_1Placeholder*/
_output_shapes
:���������dd*
dtype0*$
shape:���������dd
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cat_block/conv2d/kernelcat_block/conv2d/bias#cat_block/batch_normalization/gamma"cat_block/batch_normalization/beta)cat_block/batch_normalization/moving_mean-cat_block/batch_normalization/moving_variancecat_block_1/conv2d_1/kernelcat_block_1/conv2d_1/bias'cat_block_1/batch_normalization_1/gamma&cat_block_1/batch_normalization_1/beta-cat_block_1/batch_normalization_1/moving_mean1cat_block_1/batch_normalization_1/moving_variancecat_block_2/conv2d_2/kernelcat_block_2/conv2d_2/bias'cat_block_2/batch_normalization_2/gamma&cat_block_2/batch_normalization_2/beta-cat_block_2/batch_normalization_2/moving_mean1cat_block_2/batch_normalization_2/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_107286

NoOpNoOp
Վ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
cnn1
	cnn2

cnn3
flatten
	connected
dropout

classifier
	optimizer

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21*
z
0
1
2
3
4
5
6
7
8
9
10
 11
#12
$13
%14
&15*
* 
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

,trace_0
-trace_1* 

.trace_0
/trace_1* 
* 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6conv
7bn
8pooling*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?conv
@bn
Apooling*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Hconv
Ibn
Jpooling*
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses* 
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

#kernel
$bias*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator* 
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

%kernel
&bias*
�
d
_variables
e_iterations
f_learning_rate
g_index_dict
h
_momentums
i_velocities
j_update_step_xla*

kserving_default* 
WQ
VARIABLE_VALUEcat_block/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcat_block/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#cat_block/batch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE"cat_block/batch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUE)cat_block/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-cat_block/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEcat_block_1/conv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEcat_block_1/conv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE'cat_block_1/batch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&cat_block_1/batch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-cat_block_1/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1cat_block_1/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEcat_block_2/conv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEcat_block_2/conv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUE'cat_block_2/batch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE&cat_block_2/batch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-cat_block_2/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1cat_block_2/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
2
3
!4
"5*
5
0
	1

2
3
4
5
6*

l0
m1*
* 
* 
* 
* 
* 
* 
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

strace_0
ttrace_1* 

utrace_0
vtrace_1* 
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

kernel
bias
 }_jit_compiled_convolution_op*
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
.
0
1
2
 3
!4
"5*
 
0
1
2
 3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
 beta
!moving_mean
"moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

%0
&1*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
�
e0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_15* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*

0
1*

60
71
82*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*
* 
* 
* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
1*

?0
@1
A2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
 
0
1
2
3*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

!0
"1*

H0
I1
J2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
 
0
 1
!2
"3*

0
 1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
ic
VARIABLE_VALUEAdam/m/cat_block/conv2d/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/cat_block/conv2d/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/cat_block/conv2d/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/cat_block/conv2d/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/m/cat_block/batch_normalization/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE*Adam/v/cat_block/batch_normalization/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/m/cat_block/batch_normalization/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE)Adam/v/cat_block/batch_normalization/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/cat_block_1/conv2d_1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/cat_block_1/conv2d_1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/cat_block_1/conv2d_1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/cat_block_1/conv2d_1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/m/cat_block_1/batch_normalization_1/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/cat_block_1/batch_normalization_1/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/cat_block_1/batch_normalization_1/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/cat_block_1/batch_normalization_1/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/cat_block_2/conv2d_2/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/cat_block_2/conv2d_2/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/cat_block_2/conv2d_2/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/cat_block_2/conv2d_2/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/m/cat_block_2/batch_normalization_2/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE.Adam/v/cat_block_2/batch_normalization_2/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/m/cat_block_2/batch_normalization_2/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE-Adam/v/cat_block_2/batch_normalization_2/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_1/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_1/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_1/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_1/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

!0
"1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecat_block/conv2d/kernelcat_block/conv2d/bias#cat_block/batch_normalization/gamma"cat_block/batch_normalization/beta)cat_block/batch_normalization/moving_mean-cat_block/batch_normalization/moving_variancecat_block_1/conv2d_1/kernelcat_block_1/conv2d_1/bias'cat_block_1/batch_normalization_1/gamma&cat_block_1/batch_normalization_1/beta-cat_block_1/batch_normalization_1/moving_mean1cat_block_1/batch_normalization_1/moving_variancecat_block_2/conv2d_2/kernelcat_block_2/conv2d_2/bias'cat_block_2/batch_normalization_2/gamma&cat_block_2/batch_normalization_2/beta-cat_block_2/batch_normalization_2/moving_mean1cat_block_2/batch_normalization_2/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationlearning_rateAdam/m/cat_block/conv2d/kernelAdam/v/cat_block/conv2d/kernelAdam/m/cat_block/conv2d/biasAdam/v/cat_block/conv2d/bias*Adam/m/cat_block/batch_normalization/gamma*Adam/v/cat_block/batch_normalization/gamma)Adam/m/cat_block/batch_normalization/beta)Adam/v/cat_block/batch_normalization/beta"Adam/m/cat_block_1/conv2d_1/kernel"Adam/v/cat_block_1/conv2d_1/kernel Adam/m/cat_block_1/conv2d_1/bias Adam/v/cat_block_1/conv2d_1/bias.Adam/m/cat_block_1/batch_normalization_1/gamma.Adam/v/cat_block_1/batch_normalization_1/gamma-Adam/m/cat_block_1/batch_normalization_1/beta-Adam/v/cat_block_1/batch_normalization_1/beta"Adam/m/cat_block_2/conv2d_2/kernel"Adam/v/cat_block_2/conv2d_2/kernel Adam/m/cat_block_2/conv2d_2/bias Adam/v/cat_block_2/conv2d_2/bias.Adam/m/cat_block_2/batch_normalization_2/gamma.Adam/v/cat_block_2/batch_normalization_2/gamma-Adam/m/cat_block_2/batch_normalization_2/beta-Adam/v/cat_block_2/batch_normalization_2/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcountConst*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_108300
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecat_block/conv2d/kernelcat_block/conv2d/bias#cat_block/batch_normalization/gamma"cat_block/batch_normalization/beta)cat_block/batch_normalization/moving_mean-cat_block/batch_normalization/moving_variancecat_block_1/conv2d_1/kernelcat_block_1/conv2d_1/bias'cat_block_1/batch_normalization_1/gamma&cat_block_1/batch_normalization_1/beta-cat_block_1/batch_normalization_1/moving_mean1cat_block_1/batch_normalization_1/moving_variancecat_block_2/conv2d_2/kernelcat_block_2/conv2d_2/bias'cat_block_2/batch_normalization_2/gamma&cat_block_2/batch_normalization_2/beta-cat_block_2/batch_normalization_2/moving_mean1cat_block_2/batch_normalization_2/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias	iterationlearning_rateAdam/m/cat_block/conv2d/kernelAdam/v/cat_block/conv2d/kernelAdam/m/cat_block/conv2d/biasAdam/v/cat_block/conv2d/bias*Adam/m/cat_block/batch_normalization/gamma*Adam/v/cat_block/batch_normalization/gamma)Adam/m/cat_block/batch_normalization/beta)Adam/v/cat_block/batch_normalization/beta"Adam/m/cat_block_1/conv2d_1/kernel"Adam/v/cat_block_1/conv2d_1/kernel Adam/m/cat_block_1/conv2d_1/bias Adam/v/cat_block_1/conv2d_1/bias.Adam/m/cat_block_1/batch_normalization_1/gamma.Adam/v/cat_block_1/batch_normalization_1/gamma-Adam/m/cat_block_1/batch_normalization_1/beta-Adam/v/cat_block_1/batch_normalization_1/beta"Adam/m/cat_block_2/conv2d_2/kernel"Adam/v/cat_block_2/conv2d_2/kernel Adam/m/cat_block_2/conv2d_2/bias Adam/v/cat_block_2/conv2d_2/bias.Adam/m/cat_block_2/batch_normalization_2/gamma.Adam/v/cat_block_2/batch_normalization_2/gamma-Adam/m/cat_block_2/batch_normalization_2/beta-Adam/v/cat_block_2/batch_normalization_2/betaAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/dense_1/kernelAdam/v/dense_1/kernelAdam/m/dense_1/biasAdam/v/dense_1/biastotal_1count_1totalcount*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_108489��
�
D
(__inference_flatten_layer_call_fn_107629

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106815b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������

�:X T
0
_output_shapes
:���������

�
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_107366
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:
"
_user_specified_name
gradient
�(
�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107598	
inputC
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dinput&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_2/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:���������

��
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
0
_output_shapes
:����������

_user_specified_nameinput
�
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_106541

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_107286
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:���

unknown_18:	�

unknown_19:	�

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_106474o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107282:&"
 
_user_specified_name107280:&"
 
_user_specified_name107278:&"
 
_user_specified_name107276:&"
 
_user_specified_name107274:&"
 
_user_specified_name107272:&"
 
_user_specified_name107270:&"
 
_user_specified_name107268:&"
 
_user_specified_name107266:&"
 
_user_specified_name107264:&"
 
_user_specified_name107262:&"
 
_user_specified_name107260:&
"
 
_user_specified_name107258:&	"
 
_user_specified_name107256:&"
 
_user_specified_name107254:&"
 
_user_specified_name107252:&"
 
_user_specified_name107250:&"
 
_user_specified_name107248:&"
 
_user_specified_name107246:&"
 
_user_specified_name107244:&"
 
_user_specified_name107242:&"
 
_user_specified_name107240:X T
/
_output_shapes
:���������dd
!
_user_specified_name	input_1
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_106636

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_106492

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_107677

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_106930	
inputB
'conv2d_1_conv2d_readvariableop_resource:@�7
(conv2d_1_biasadd_readvariableop_resource:	�<
-batch_normalization_1_readvariableop_resource:	�>
/batch_normalization_1_readvariableop_1_resource:	�M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������//��
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������//�:�:�:�:�:*
epsilon%o�:*
is_training( �
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������11@: : : : : : 2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������11@

_user_specified_nameinput
�-
�
C__inference_cat_net_layer_call_and_return_conditional_losses_107001
input_1*
cat_block_106892:@
cat_block_106894:@
cat_block_106896:@
cat_block_106898:@
cat_block_106900:@
cat_block_106902:@-
cat_block_1_106931:@�!
cat_block_1_106933:	�!
cat_block_1_106935:	�!
cat_block_1_106937:	�!
cat_block_1_106939:	�!
cat_block_1_106941:	�.
cat_block_2_106970:��!
cat_block_2_106972:	�!
cat_block_2_106974:	�!
cat_block_2_106976:	�!
cat_block_2_106978:	�!
cat_block_2_106980:	�!
dense_106984:���
dense_106986:	�!
dense_1_106995:	�
dense_1_106997:
identity��!cat_block/StatefulPartitionedCall�#cat_block_1/StatefulPartitionedCall�#cat_block_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
!cat_block/StatefulPartitionedCallStatefulPartitionedCallinput_1cat_block_106892cat_block_106894cat_block_106896cat_block_106898cat_block_106900cat_block_106902*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������11@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_cat_block_layer_call_and_return_conditional_losses_106891�
#cat_block_1/StatefulPartitionedCallStatefulPartitionedCall*cat_block/StatefulPartitionedCall:output:0cat_block_1_106931cat_block_1_106933cat_block_1_106935cat_block_1_106937cat_block_1_106939cat_block_1_106941*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_1_layer_call_and_return_conditional_losses_106930�
#cat_block_2/StatefulPartitionedCallStatefulPartitionedCall,cat_block_1/StatefulPartitionedCall:output:0cat_block_2_106970cat_block_2_106972cat_block_2_106974cat_block_2_106976cat_block_2_106978cat_block_2_106980*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_2_layer_call_and_return_conditional_losses_106969�
flatten/PartitionedCallPartitionedCall,cat_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106815�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_106984dense_106986*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_106827�
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106993�
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_106995dense_1_106997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_106856w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^cat_block/StatefulPartitionedCall$^cat_block_1/StatefulPartitionedCall$^cat_block_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������dd: : : : : : : : : : : : : : : : : : : : : : 2F
!cat_block/StatefulPartitionedCall!cat_block/StatefulPartitionedCall2J
#cat_block_1/StatefulPartitionedCall#cat_block_1/StatefulPartitionedCall2J
#cat_block_2/StatefulPartitionedCall#cat_block_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:&"
 
_user_specified_name106997:&"
 
_user_specified_name106995:&"
 
_user_specified_name106986:&"
 
_user_specified_name106984:&"
 
_user_specified_name106980:&"
 
_user_specified_name106978:&"
 
_user_specified_name106976:&"
 
_user_specified_name106974:&"
 
_user_specified_name106972:&"
 
_user_specified_name106970:&"
 
_user_specified_name106941:&"
 
_user_specified_name106939:&
"
 
_user_specified_name106937:&	"
 
_user_specified_name106935:&"
 
_user_specified_name106933:&"
 
_user_specified_name106931:&"
 
_user_specified_name106902:&"
 
_user_specified_name106900:&"
 
_user_specified_name106898:&"
 
_user_specified_name106896:&"
 
_user_specified_name106894:&"
 
_user_specified_name106892:X T
/
_output_shapes
:���������dd
!
_user_specified_name	input_1
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_107702

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�(
�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107512	
inputB
'conv2d_1_conv2d_readvariableop_resource:@�7
(conv2d_1_biasadd_readvariableop_resource:	�<
-batch_normalization_1_readvariableop_resource:	�>
/batch_normalization_1_readvariableop_1_resource:	�M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������//��
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������//�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������11@: : : : : : 2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������11@

_user_specified_nameinput
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_106654

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_1_layer_call_fn_107800

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_106582�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107796:&"
 
_user_specified_name107794:&"
 
_user_specified_name107792:&"
 
_user_specified_name107790:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
P
#__inference__update_step_xla_107361
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: *
	_noinline(:($
"
_user_specified_name
variable:I E

_output_shapes
:	�
"
_user_specified_name
gradient
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_107635

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� d  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������

�:X T
0
_output_shapes
:���������

�
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_106827

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107836

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�&
�
E__inference_cat_block_layer_call_and_return_conditional_losses_107426	
input?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������bb@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������bb@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:���������11@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:���������11@�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������dd: : : : : : 2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������dd

_user_specified_nameinput
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_106582

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
,__inference_cat_block_1_layer_call_fn_107469	
input"
unknown:@�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_1_layer_call_and_return_conditional_losses_106757x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������11@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107465:&"
 
_user_specified_name107463:&"
 
_user_specified_name107461:&"
 
_user_specified_name107459:&"
 
_user_specified_name107457:&"
 
_user_specified_name107455:V R
/
_output_shapes
:���������11@

_user_specified_nameinput
�
L
#__inference__update_step_xla_107321
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
�
(__inference_dense_1_layer_call_fn_107691

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_106856o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107687:&"
 
_user_specified_name107685:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_106969	
inputC
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dinput&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_2/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:���������

��
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
0
_output_shapes
:����������

_user_specified_nameinput
�
D
(__inference_dropout_layer_call_fn_107665

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106993a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107746

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�

�
4__inference_batch_normalization_layer_call_fn_107715

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_106492�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107711:&"
 
_user_specified_name107709:&"
 
_user_specified_name107707:&"
 
_user_specified_name107705:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_106685

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_107356
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_106613

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
,__inference_cat_block_2_layer_call_fn_107572	
input#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_2_layer_call_and_return_conditional_losses_106969x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������

�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107568:&"
 
_user_specified_name107566:&"
 
_user_specified_name107564:&"
 
_user_specified_name107562:&"
 
_user_specified_name107560:&"
 
_user_specified_name107558:W S
0
_output_shapes
:����������

_user_specified_nameinput
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107818

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
E__inference_cat_block_layer_call_and_return_conditional_losses_107452	
input?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������bb@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������bb@:@:@:@:@:*
epsilon%o�:*
is_training( �
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:���������11@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:���������11@�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������dd: : : : : : 2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������dd

_user_specified_nameinput
ً
�
!__inference__wrapped_model_106474
input_1Q
7cat_net_cat_block_conv2d_conv2d_readvariableop_resource:@F
8cat_net_cat_block_conv2d_biasadd_readvariableop_resource:@K
=cat_net_cat_block_batch_normalization_readvariableop_resource:@M
?cat_net_cat_block_batch_normalization_readvariableop_1_resource:@\
Ncat_net_cat_block_batch_normalization_fusedbatchnormv3_readvariableop_resource:@^
Pcat_net_cat_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@V
;cat_net_cat_block_1_conv2d_1_conv2d_readvariableop_resource:@�K
<cat_net_cat_block_1_conv2d_1_biasadd_readvariableop_resource:	�P
Acat_net_cat_block_1_batch_normalization_1_readvariableop_resource:	�R
Ccat_net_cat_block_1_batch_normalization_1_readvariableop_1_resource:	�a
Rcat_net_cat_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�c
Tcat_net_cat_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�W
;cat_net_cat_block_2_conv2d_2_conv2d_readvariableop_resource:��K
<cat_net_cat_block_2_conv2d_2_biasadd_readvariableop_resource:	�P
Acat_net_cat_block_2_batch_normalization_2_readvariableop_resource:	�R
Ccat_net_cat_block_2_batch_normalization_2_readvariableop_1_resource:	�a
Rcat_net_cat_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�c
Tcat_net_cat_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�A
,cat_net_dense_matmul_readvariableop_resource:���<
-cat_net_dense_biasadd_readvariableop_resource:	�A
.cat_net_dense_1_matmul_readvariableop_resource:	�=
/cat_net_dense_1_biasadd_readvariableop_resource:
identity��Ecat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp�Gcat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�4cat_net/cat_block/batch_normalization/ReadVariableOp�6cat_net/cat_block/batch_normalization/ReadVariableOp_1�/cat_net/cat_block/conv2d/BiasAdd/ReadVariableOp�.cat_net/cat_block/conv2d/Conv2D/ReadVariableOp�Icat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Kcat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�8cat_net/cat_block_1/batch_normalization_1/ReadVariableOp�:cat_net/cat_block_1/batch_normalization_1/ReadVariableOp_1�3cat_net/cat_block_1/conv2d_1/BiasAdd/ReadVariableOp�2cat_net/cat_block_1/conv2d_1/Conv2D/ReadVariableOp�Icat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�Kcat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�8cat_net/cat_block_2/batch_normalization_2/ReadVariableOp�:cat_net/cat_block_2/batch_normalization_2/ReadVariableOp_1�3cat_net/cat_block_2/conv2d_2/BiasAdd/ReadVariableOp�2cat_net/cat_block_2/conv2d_2/Conv2D/ReadVariableOp�$cat_net/dense/BiasAdd/ReadVariableOp�#cat_net/dense/MatMul/ReadVariableOp�&cat_net/dense_1/BiasAdd/ReadVariableOp�%cat_net/dense_1/MatMul/ReadVariableOp�
.cat_net/cat_block/conv2d/Conv2D/ReadVariableOpReadVariableOp7cat_net_cat_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
cat_net/cat_block/conv2d/Conv2DConv2Dinput_16cat_net/cat_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@*
paddingVALID*
strides
�
/cat_net/cat_block/conv2d/BiasAdd/ReadVariableOpReadVariableOp8cat_net_cat_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 cat_net/cat_block/conv2d/BiasAddBiasAdd(cat_net/cat_block/conv2d/Conv2D:output:07cat_net/cat_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@�
cat_net/cat_block/conv2d/ReluRelu)cat_net/cat_block/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������bb@�
4cat_net/cat_block/batch_normalization/ReadVariableOpReadVariableOp=cat_net_cat_block_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
6cat_net/cat_block/batch_normalization/ReadVariableOp_1ReadVariableOp?cat_net_cat_block_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
Ecat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpNcat_net_cat_block_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
Gcat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPcat_net_cat_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
6cat_net/cat_block/batch_normalization/FusedBatchNormV3FusedBatchNormV3+cat_net/cat_block/conv2d/Relu:activations:0<cat_net/cat_block/batch_normalization/ReadVariableOp:value:0>cat_net/cat_block/batch_normalization/ReadVariableOp_1:value:0Mcat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ocat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������bb@:@:@:@:@:*
epsilon%o�:*
is_training( �
'cat_net/cat_block/max_pooling2d/MaxPoolMaxPool:cat_net/cat_block/batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:���������11@*
ksize
*
paddingVALID*
strides
�
2cat_net/cat_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;cat_net_cat_block_1_conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
#cat_net/cat_block_1/conv2d_1/Conv2DConv2D0cat_net/cat_block/max_pooling2d/MaxPool:output:0:cat_net/cat_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�*
paddingVALID*
strides
�
3cat_net/cat_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<cat_net_cat_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$cat_net/cat_block_1/conv2d_1/BiasAddBiasAdd,cat_net/cat_block_1/conv2d_1/Conv2D:output:0;cat_net/cat_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//��
!cat_net/cat_block_1/conv2d_1/ReluRelu-cat_net/cat_block_1/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������//��
8cat_net/cat_block_1/batch_normalization_1/ReadVariableOpReadVariableOpAcat_net_cat_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:cat_net/cat_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpCcat_net_cat_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Icat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpRcat_net_cat_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Kcat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcat_net_cat_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
:cat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3/cat_net/cat_block_1/conv2d_1/Relu:activations:0@cat_net/cat_block_1/batch_normalization_1/ReadVariableOp:value:0Bcat_net/cat_block_1/batch_normalization_1/ReadVariableOp_1:value:0Qcat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Scat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������//�:�:�:�:�:*
epsilon%o�:*
is_training( �
+cat_net/cat_block_1/max_pooling2d_1/MaxPoolMaxPool>cat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
2cat_net/cat_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;cat_net_cat_block_2_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
#cat_net/cat_block_2/conv2d_2/Conv2DConv2D4cat_net/cat_block_1/max_pooling2d_1/MaxPool:output:0:cat_net/cat_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
3cat_net/cat_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<cat_net_cat_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$cat_net/cat_block_2/conv2d_2/BiasAddBiasAdd,cat_net/cat_block_2/conv2d_2/Conv2D:output:0;cat_net/cat_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
!cat_net/cat_block_2/conv2d_2/ReluRelu-cat_net/cat_block_2/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
8cat_net/cat_block_2/batch_normalization_2/ReadVariableOpReadVariableOpAcat_net_cat_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
:cat_net/cat_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpCcat_net_cat_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
Icat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpRcat_net_cat_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
Kcat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpTcat_net_cat_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
:cat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3/cat_net/cat_block_2/conv2d_2/Relu:activations:0@cat_net/cat_block_2/batch_normalization_2/ReadVariableOp:value:0Bcat_net/cat_block_2/batch_normalization_2/ReadVariableOp_1:value:0Qcat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Scat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
+cat_net/cat_block_2/max_pooling2d_2/MaxPoolMaxPool>cat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
f
cat_net/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� d  �
cat_net/flatten/ReshapeReshape4cat_net/cat_block_2/max_pooling2d_2/MaxPool:output:0cat_net/flatten/Const:output:0*
T0*)
_output_shapes
:������������
#cat_net/dense/MatMul/ReadVariableOpReadVariableOp,cat_net_dense_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
cat_net/dense/MatMulMatMul cat_net/flatten/Reshape:output:0+cat_net/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$cat_net/dense/BiasAdd/ReadVariableOpReadVariableOp-cat_net_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
cat_net/dense/BiasAddBiasAddcat_net/dense/MatMul:product:0,cat_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
cat_net/dense/ReluRelucat_net/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������y
cat_net/dropout/IdentityIdentity cat_net/dense/Relu:activations:0*
T0*(
_output_shapes
:�����������
%cat_net/dense_1/MatMul/ReadVariableOpReadVariableOp.cat_net_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
cat_net/dense_1/MatMulMatMul!cat_net/dropout/Identity:output:0-cat_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&cat_net/dense_1/BiasAdd/ReadVariableOpReadVariableOp/cat_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
cat_net/dense_1/BiasAddBiasAdd cat_net/dense_1/MatMul:product:0.cat_net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
cat_net/dense_1/SigmoidSigmoid cat_net/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������j
IdentityIdentitycat_net/dense_1/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:����������

NoOpNoOpF^cat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOpH^cat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_15^cat_net/cat_block/batch_normalization/ReadVariableOp7^cat_net/cat_block/batch_normalization/ReadVariableOp_10^cat_net/cat_block/conv2d/BiasAdd/ReadVariableOp/^cat_net/cat_block/conv2d/Conv2D/ReadVariableOpJ^cat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpL^cat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_19^cat_net/cat_block_1/batch_normalization_1/ReadVariableOp;^cat_net/cat_block_1/batch_normalization_1/ReadVariableOp_14^cat_net/cat_block_1/conv2d_1/BiasAdd/ReadVariableOp3^cat_net/cat_block_1/conv2d_1/Conv2D/ReadVariableOpJ^cat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpL^cat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_19^cat_net/cat_block_2/batch_normalization_2/ReadVariableOp;^cat_net/cat_block_2/batch_normalization_2/ReadVariableOp_14^cat_net/cat_block_2/conv2d_2/BiasAdd/ReadVariableOp3^cat_net/cat_block_2/conv2d_2/Conv2D/ReadVariableOp%^cat_net/dense/BiasAdd/ReadVariableOp$^cat_net/dense/MatMul/ReadVariableOp'^cat_net/dense_1/BiasAdd/ReadVariableOp&^cat_net/dense_1/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������dd: : : : : : : : : : : : : : : : : : : : : : 2�
Gcat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Gcat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_12�
Ecat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOpEcat_net/cat_block/batch_normalization/FusedBatchNormV3/ReadVariableOp2p
6cat_net/cat_block/batch_normalization/ReadVariableOp_16cat_net/cat_block/batch_normalization/ReadVariableOp_12l
4cat_net/cat_block/batch_normalization/ReadVariableOp4cat_net/cat_block/batch_normalization/ReadVariableOp2b
/cat_net/cat_block/conv2d/BiasAdd/ReadVariableOp/cat_net/cat_block/conv2d/BiasAdd/ReadVariableOp2`
.cat_net/cat_block/conv2d/Conv2D/ReadVariableOp.cat_net/cat_block/conv2d/Conv2D/ReadVariableOp2�
Kcat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Kcat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12�
Icat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpIcat_net/cat_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2x
:cat_net/cat_block_1/batch_normalization_1/ReadVariableOp_1:cat_net/cat_block_1/batch_normalization_1/ReadVariableOp_12t
8cat_net/cat_block_1/batch_normalization_1/ReadVariableOp8cat_net/cat_block_1/batch_normalization_1/ReadVariableOp2j
3cat_net/cat_block_1/conv2d_1/BiasAdd/ReadVariableOp3cat_net/cat_block_1/conv2d_1/BiasAdd/ReadVariableOp2h
2cat_net/cat_block_1/conv2d_1/Conv2D/ReadVariableOp2cat_net/cat_block_1/conv2d_1/Conv2D/ReadVariableOp2�
Kcat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Kcat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12�
Icat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpIcat_net/cat_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2x
:cat_net/cat_block_2/batch_normalization_2/ReadVariableOp_1:cat_net/cat_block_2/batch_normalization_2/ReadVariableOp_12t
8cat_net/cat_block_2/batch_normalization_2/ReadVariableOp8cat_net/cat_block_2/batch_normalization_2/ReadVariableOp2j
3cat_net/cat_block_2/conv2d_2/BiasAdd/ReadVariableOp3cat_net/cat_block_2/conv2d_2/BiasAdd/ReadVariableOp2h
2cat_net/cat_block_2/conv2d_2/Conv2D/ReadVariableOp2cat_net/cat_block_2/conv2d_2/Conv2D/ReadVariableOp2L
$cat_net/dense/BiasAdd/ReadVariableOp$cat_net/dense/BiasAdd/ReadVariableOp2J
#cat_net/dense/MatMul/ReadVariableOp#cat_net/dense/MatMul/ReadVariableOp2P
&cat_net/dense_1/BiasAdd/ReadVariableOp&cat_net/dense_1/BiasAdd/ReadVariableOp2N
%cat_net/dense_1/MatMul/ReadVariableOp%cat_net/dense_1/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
/
_output_shapes
:���������dd
!
_user_specified_name	input_1
�&
�
E__inference_cat_block_layer_call_and_return_conditional_losses_106718	
input?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������bb@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������bb@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:���������11@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:���������11@�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������dd: : : : : : 2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������dd

_user_specified_nameinput
�(
�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_106757	
inputB
'conv2d_1_conv2d_readvariableop_resource:@�7
(conv2d_1_biasadd_readvariableop_resource:	�<
-batch_normalization_1_readvariableop_resource:	�>
/batch_normalization_1_readvariableop_1_resource:	�M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������//��
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������//�:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������11@: : : : : : 2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������11@

_user_specified_nameinput
�
a
(__inference_dropout_layer_call_fn_107660

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106844p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_flatten_layer_call_and_return_conditional_losses_106815

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� d  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������

�:X T
0
_output_shapes
:���������

�
 
_user_specified_nameinputs
�

�
,__inference_cat_block_1_layer_call_fn_107486	
input"
unknown:@�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_1_layer_call_and_return_conditional_losses_106930x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������11@: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107482:&"
 
_user_specified_name107480:&"
 
_user_specified_name107478:&"
 
_user_specified_name107476:&"
 
_user_specified_name107474:&"
 
_user_specified_name107472:V R
/
_output_shapes
:���������11@

_user_specified_nameinput
�
L
#__inference__update_step_xla_107316
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
L
#__inference__update_step_xla_107336
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107764

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_106993

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_cat_block_layer_call_and_return_conditional_losses_106891	
input?
%conv2d_conv2d_readvariableop_resource:@4
&conv2d_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
conv2d/Conv2DConv2Dinput$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������bb@f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������bb@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������bb@:@:@:@:@:*
epsilon%o�:*
is_training( �
max_pooling2d/MaxPoolMaxPool(batch_normalization/FusedBatchNormV3:y:0*/
_output_shapes
:���������11@*
ksize
*
paddingVALID*
strides
u
IdentityIdentitymax_pooling2d/MaxPool:output:0^NoOp*
T0*/
_output_shapes
:���������11@�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������dd: : : : : : 2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������dd

_user_specified_nameinput
�

�
6__inference_batch_normalization_2_layer_call_fn_107859

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_106636�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107855:&"
 
_user_specified_name107853:&"
 
_user_specified_name107851:&"
 
_user_specified_name107849:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_2_layer_call_fn_107913

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_106685�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_107326
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�
K
#__inference__update_step_xla_107306
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
�
K
#__inference__update_step_xla_107296
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
�

b
C__inference_dropout_layer_call_and_return_conditional_losses_106844

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
� 
�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107624	
inputC
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dinput&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
is_training( �
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_2/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:���������

��
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
0
_output_shapes
:����������

_user_specified_nameinput
�

�
A__inference_dense_layer_call_and_return_conditional_losses_107655

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
K
#__inference__update_step_xla_107301
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:($
"
_user_specified_name
variable:D @

_output_shapes
:@
"
_user_specified_name
gradient
��
�<
__inference__traced_save_108300
file_prefixH
.read_disablecopyonread_cat_block_conv2d_kernel:@<
.read_1_disablecopyonread_cat_block_conv2d_bias:@J
<read_2_disablecopyonread_cat_block_batch_normalization_gamma:@I
;read_3_disablecopyonread_cat_block_batch_normalization_beta:@P
Bread_4_disablecopyonread_cat_block_batch_normalization_moving_mean:@T
Fread_5_disablecopyonread_cat_block_batch_normalization_moving_variance:@O
4read_6_disablecopyonread_cat_block_1_conv2d_1_kernel:@�A
2read_7_disablecopyonread_cat_block_1_conv2d_1_bias:	�O
@read_8_disablecopyonread_cat_block_1_batch_normalization_1_gamma:	�N
?read_9_disablecopyonread_cat_block_1_batch_normalization_1_beta:	�V
Gread_10_disablecopyonread_cat_block_1_batch_normalization_1_moving_mean:	�Z
Kread_11_disablecopyonread_cat_block_1_batch_normalization_1_moving_variance:	�Q
5read_12_disablecopyonread_cat_block_2_conv2d_2_kernel:��B
3read_13_disablecopyonread_cat_block_2_conv2d_2_bias:	�P
Aread_14_disablecopyonread_cat_block_2_batch_normalization_2_gamma:	�O
@read_15_disablecopyonread_cat_block_2_batch_normalization_2_beta:	�V
Gread_16_disablecopyonread_cat_block_2_batch_normalization_2_moving_mean:	�Z
Kread_17_disablecopyonread_cat_block_2_batch_normalization_2_moving_variance:	�;
&read_18_disablecopyonread_dense_kernel:���3
$read_19_disablecopyonread_dense_bias:	�;
(read_20_disablecopyonread_dense_1_kernel:	�4
&read_21_disablecopyonread_dense_1_bias:-
#read_22_disablecopyonread_iteration:	 1
'read_23_disablecopyonread_learning_rate: R
8read_24_disablecopyonread_adam_m_cat_block_conv2d_kernel:@R
8read_25_disablecopyonread_adam_v_cat_block_conv2d_kernel:@D
6read_26_disablecopyonread_adam_m_cat_block_conv2d_bias:@D
6read_27_disablecopyonread_adam_v_cat_block_conv2d_bias:@R
Dread_28_disablecopyonread_adam_m_cat_block_batch_normalization_gamma:@R
Dread_29_disablecopyonread_adam_v_cat_block_batch_normalization_gamma:@Q
Cread_30_disablecopyonread_adam_m_cat_block_batch_normalization_beta:@Q
Cread_31_disablecopyonread_adam_v_cat_block_batch_normalization_beta:@W
<read_32_disablecopyonread_adam_m_cat_block_1_conv2d_1_kernel:@�W
<read_33_disablecopyonread_adam_v_cat_block_1_conv2d_1_kernel:@�I
:read_34_disablecopyonread_adam_m_cat_block_1_conv2d_1_bias:	�I
:read_35_disablecopyonread_adam_v_cat_block_1_conv2d_1_bias:	�W
Hread_36_disablecopyonread_adam_m_cat_block_1_batch_normalization_1_gamma:	�W
Hread_37_disablecopyonread_adam_v_cat_block_1_batch_normalization_1_gamma:	�V
Gread_38_disablecopyonread_adam_m_cat_block_1_batch_normalization_1_beta:	�V
Gread_39_disablecopyonread_adam_v_cat_block_1_batch_normalization_1_beta:	�X
<read_40_disablecopyonread_adam_m_cat_block_2_conv2d_2_kernel:��X
<read_41_disablecopyonread_adam_v_cat_block_2_conv2d_2_kernel:��I
:read_42_disablecopyonread_adam_m_cat_block_2_conv2d_2_bias:	�I
:read_43_disablecopyonread_adam_v_cat_block_2_conv2d_2_bias:	�W
Hread_44_disablecopyonread_adam_m_cat_block_2_batch_normalization_2_gamma:	�W
Hread_45_disablecopyonread_adam_v_cat_block_2_batch_normalization_2_gamma:	�V
Gread_46_disablecopyonread_adam_m_cat_block_2_batch_normalization_2_beta:	�V
Gread_47_disablecopyonread_adam_v_cat_block_2_batch_normalization_2_beta:	�B
-read_48_disablecopyonread_adam_m_dense_kernel:���B
-read_49_disablecopyonread_adam_v_dense_kernel:���:
+read_50_disablecopyonread_adam_m_dense_bias:	�:
+read_51_disablecopyonread_adam_v_dense_bias:	�B
/read_52_disablecopyonread_adam_m_dense_1_kernel:	�B
/read_53_disablecopyonread_adam_v_dense_1_kernel:	�;
-read_54_disablecopyonread_adam_m_dense_1_bias:;
-read_55_disablecopyonread_adam_v_dense_1_bias:+
!read_56_disablecopyonread_total_1: +
!read_57_disablecopyonread_count_1: )
read_58_disablecopyonread_total: )
read_59_disablecopyonread_count: 
savev2_const
identity_121��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
Read/DisableCopyOnReadDisableCopyOnRead.read_disablecopyonread_cat_block_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp.read_disablecopyonread_cat_block_conv2d_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_1/DisableCopyOnReadDisableCopyOnRead.read_1_disablecopyonread_cat_block_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp.read_1_disablecopyonread_cat_block_conv2d_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_2/DisableCopyOnReadDisableCopyOnRead<read_2_disablecopyonread_cat_block_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp<read_2_disablecopyonread_cat_block_batch_normalization_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_3/DisableCopyOnReadDisableCopyOnRead;read_3_disablecopyonread_cat_block_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp;read_3_disablecopyonread_cat_block_batch_normalization_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_4/DisableCopyOnReadDisableCopyOnReadBread_4_disablecopyonread_cat_block_batch_normalization_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpBread_4_disablecopyonread_cat_block_batch_normalization_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_5/DisableCopyOnReadDisableCopyOnReadFread_5_disablecopyonread_cat_block_batch_normalization_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpFread_5_disablecopyonread_cat_block_batch_normalization_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_6/DisableCopyOnReadDisableCopyOnRead4read_6_disablecopyonread_cat_block_1_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp4read_6_disablecopyonread_cat_block_1_conv2d_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0w
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_7/DisableCopyOnReadDisableCopyOnRead2read_7_disablecopyonread_cat_block_1_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp2read_7_disablecopyonread_cat_block_1_conv2d_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_8/DisableCopyOnReadDisableCopyOnRead@read_8_disablecopyonread_cat_block_1_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp@read_8_disablecopyonread_cat_block_1_batch_normalization_1_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_9/DisableCopyOnReadDisableCopyOnRead?read_9_disablecopyonread_cat_block_1_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp?read_9_disablecopyonread_cat_block_1_batch_normalization_1_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnReadGread_10_disablecopyonread_cat_block_1_batch_normalization_1_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOpGread_10_disablecopyonread_cat_block_1_batch_normalization_1_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_11/DisableCopyOnReadDisableCopyOnReadKread_11_disablecopyonread_cat_block_1_batch_normalization_1_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOpKread_11_disablecopyonread_cat_block_1_batch_normalization_1_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnRead5read_12_disablecopyonread_cat_block_2_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp5read_12_disablecopyonread_cat_block_2_conv2d_2_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_13/DisableCopyOnReadDisableCopyOnRead3read_13_disablecopyonread_cat_block_2_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp3read_13_disablecopyonread_cat_block_2_conv2d_2_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnReadAread_14_disablecopyonread_cat_block_2_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOpAread_14_disablecopyonread_cat_block_2_batch_normalization_2_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_15/DisableCopyOnReadDisableCopyOnRead@read_15_disablecopyonread_cat_block_2_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp@read_15_disablecopyonread_cat_block_2_batch_normalization_2_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnReadGread_16_disablecopyonread_cat_block_2_batch_normalization_2_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOpGread_16_disablecopyonread_cat_block_2_batch_normalization_2_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_17/DisableCopyOnReadDisableCopyOnReadKread_17_disablecopyonread_cat_block_2_batch_normalization_2_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOpKread_17_disablecopyonread_cat_block_2_batch_normalization_2_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_18/DisableCopyOnReadDisableCopyOnRead&read_18_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp&read_18_disablecopyonread_dense_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0r
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���h
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*!
_output_shapes
:���y
Read_19/DisableCopyOnReadDisableCopyOnRead$read_19_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp$read_19_disablecopyonread_dense_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_20/DisableCopyOnReadDisableCopyOnRead(read_20_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp(read_20_disablecopyonread_dense_1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_21/DisableCopyOnReadDisableCopyOnRead&read_21_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp&read_21_disablecopyonread_dense_1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_iteration^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_learning_rate^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnRead8read_24_disablecopyonread_adam_m_cat_block_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp8read_24_disablecopyonread_adam_m_cat_block_conv2d_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_25/DisableCopyOnReadDisableCopyOnRead8read_25_disablecopyonread_adam_v_cat_block_conv2d_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp8read_25_disablecopyonread_adam_v_cat_block_conv2d_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_26/DisableCopyOnReadDisableCopyOnRead6read_26_disablecopyonread_adam_m_cat_block_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp6read_26_disablecopyonread_adam_m_cat_block_conv2d_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_27/DisableCopyOnReadDisableCopyOnRead6read_27_disablecopyonread_adam_v_cat_block_conv2d_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp6read_27_disablecopyonread_adam_v_cat_block_conv2d_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_28/DisableCopyOnReadDisableCopyOnReadDread_28_disablecopyonread_adam_m_cat_block_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOpDread_28_disablecopyonread_adam_m_cat_block_batch_normalization_gamma^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_29/DisableCopyOnReadDisableCopyOnReadDread_29_disablecopyonread_adam_v_cat_block_batch_normalization_gamma"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOpDread_29_disablecopyonread_adam_v_cat_block_batch_normalization_gamma^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_30/DisableCopyOnReadDisableCopyOnReadCread_30_disablecopyonread_adam_m_cat_block_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOpCread_30_disablecopyonread_adam_m_cat_block_batch_normalization_beta^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_31/DisableCopyOnReadDisableCopyOnReadCread_31_disablecopyonread_adam_v_cat_block_batch_normalization_beta"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOpCread_31_disablecopyonread_adam_v_cat_block_batch_normalization_beta^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_32/DisableCopyOnReadDisableCopyOnRead<read_32_disablecopyonread_adam_m_cat_block_1_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp<read_32_disablecopyonread_adam_m_cat_block_1_conv2d_1_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_33/DisableCopyOnReadDisableCopyOnRead<read_33_disablecopyonread_adam_v_cat_block_1_conv2d_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp<read_33_disablecopyonread_adam_v_cat_block_1_conv2d_1_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_34/DisableCopyOnReadDisableCopyOnRead:read_34_disablecopyonread_adam_m_cat_block_1_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp:read_34_disablecopyonread_adam_m_cat_block_1_conv2d_1_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_35/DisableCopyOnReadDisableCopyOnRead:read_35_disablecopyonread_adam_v_cat_block_1_conv2d_1_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp:read_35_disablecopyonread_adam_v_cat_block_1_conv2d_1_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_36/DisableCopyOnReadDisableCopyOnReadHread_36_disablecopyonread_adam_m_cat_block_1_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOpHread_36_disablecopyonread_adam_m_cat_block_1_batch_normalization_1_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_37/DisableCopyOnReadDisableCopyOnReadHread_37_disablecopyonread_adam_v_cat_block_1_batch_normalization_1_gamma"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOpHread_37_disablecopyonread_adam_v_cat_block_1_batch_normalization_1_gamma^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_38/DisableCopyOnReadDisableCopyOnReadGread_38_disablecopyonread_adam_m_cat_block_1_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOpGread_38_disablecopyonread_adam_m_cat_block_1_batch_normalization_1_beta^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnReadGread_39_disablecopyonread_adam_v_cat_block_1_batch_normalization_1_beta"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpGread_39_disablecopyonread_adam_v_cat_block_1_batch_normalization_1_beta^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnRead<read_40_disablecopyonread_adam_m_cat_block_2_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp<read_40_disablecopyonread_adam_m_cat_block_2_conv2d_2_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_41/DisableCopyOnReadDisableCopyOnRead<read_41_disablecopyonread_adam_v_cat_block_2_conv2d_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp<read_41_disablecopyonread_adam_v_cat_block_2_conv2d_2_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_42/DisableCopyOnReadDisableCopyOnRead:read_42_disablecopyonread_adam_m_cat_block_2_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp:read_42_disablecopyonread_adam_m_cat_block_2_conv2d_2_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_43/DisableCopyOnReadDisableCopyOnRead:read_43_disablecopyonread_adam_v_cat_block_2_conv2d_2_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp:read_43_disablecopyonread_adam_v_cat_block_2_conv2d_2_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_44/DisableCopyOnReadDisableCopyOnReadHread_44_disablecopyonread_adam_m_cat_block_2_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOpHread_44_disablecopyonread_adam_m_cat_block_2_batch_normalization_2_gamma^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_45/DisableCopyOnReadDisableCopyOnReadHread_45_disablecopyonread_adam_v_cat_block_2_batch_normalization_2_gamma"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOpHread_45_disablecopyonread_adam_v_cat_block_2_batch_normalization_2_gamma^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_46/DisableCopyOnReadDisableCopyOnReadGread_46_disablecopyonread_adam_m_cat_block_2_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpGread_46_disablecopyonread_adam_m_cat_block_2_batch_normalization_2_beta^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnReadGread_47_disablecopyonread_adam_v_cat_block_2_batch_normalization_2_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOpGread_47_disablecopyonread_adam_v_cat_block_2_batch_normalization_2_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead-read_48_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp-read_48_disablecopyonread_adam_m_dense_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0r
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���h
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_49/DisableCopyOnReadDisableCopyOnRead-read_49_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp-read_49_disablecopyonread_adam_v_dense_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0r
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���h
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*!
_output_shapes
:����
Read_50/DisableCopyOnReadDisableCopyOnRead+read_50_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp+read_50_disablecopyonread_adam_m_dense_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead+read_51_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp+read_51_disablecopyonread_adam_v_dense_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnRead/read_52_disablecopyonread_adam_m_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp/read_52_disablecopyonread_adam_m_dense_1_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_53/DisableCopyOnReadDisableCopyOnRead/read_53_disablecopyonread_adam_v_dense_1_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp/read_53_disablecopyonread_adam_v_dense_1_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:	��
Read_54/DisableCopyOnReadDisableCopyOnRead-read_54_disablecopyonread_adam_m_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp-read_54_disablecopyonread_adam_m_dense_1_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_55/DisableCopyOnReadDisableCopyOnRead-read_55_disablecopyonread_adam_v_dense_1_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp-read_55_disablecopyonread_adam_v_dense_1_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_56/DisableCopyOnReadDisableCopyOnRead!read_56_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp!read_56_disablecopyonread_total_1^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_57/DisableCopyOnReadDisableCopyOnRead!read_57_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp!read_57_disablecopyonread_count_1^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_58/DisableCopyOnReadDisableCopyOnReadread_58_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpread_58_disablecopyonread_total^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_59/DisableCopyOnReadDisableCopyOnReadread_59_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpread_59_disablecopyonread_count^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *K
dtypesA
?2=	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_120Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_121IdentityIdentity_120:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_121Identity_121:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:==9

_output_shapes
: 

_user_specified_nameConst:%<!

_user_specified_namecount:%;!

_user_specified_nametotal:':#
!
_user_specified_name	count_1:'9#
!
_user_specified_name	total_1:38/
-
_user_specified_nameAdam/v/dense_1/bias:37/
-
_user_specified_nameAdam/m/dense_1/bias:561
/
_user_specified_nameAdam/v/dense_1/kernel:551
/
_user_specified_nameAdam/m/dense_1/kernel:14-
+
_user_specified_nameAdam/v/dense/bias:13-
+
_user_specified_nameAdam/m/dense/bias:32/
-
_user_specified_nameAdam/v/dense/kernel:31/
-
_user_specified_nameAdam/m/dense/kernel:M0I
G
_user_specified_name/-Adam/v/cat_block_2/batch_normalization_2/beta:M/I
G
_user_specified_name/-Adam/m/cat_block_2/batch_normalization_2/beta:N.J
H
_user_specified_name0.Adam/v/cat_block_2/batch_normalization_2/gamma:N-J
H
_user_specified_name0.Adam/m/cat_block_2/batch_normalization_2/gamma:@,<
:
_user_specified_name" Adam/v/cat_block_2/conv2d_2/bias:@+<
:
_user_specified_name" Adam/m/cat_block_2/conv2d_2/bias:B*>
<
_user_specified_name$"Adam/v/cat_block_2/conv2d_2/kernel:B)>
<
_user_specified_name$"Adam/m/cat_block_2/conv2d_2/kernel:M(I
G
_user_specified_name/-Adam/v/cat_block_1/batch_normalization_1/beta:M'I
G
_user_specified_name/-Adam/m/cat_block_1/batch_normalization_1/beta:N&J
H
_user_specified_name0.Adam/v/cat_block_1/batch_normalization_1/gamma:N%J
H
_user_specified_name0.Adam/m/cat_block_1/batch_normalization_1/gamma:@$<
:
_user_specified_name" Adam/v/cat_block_1/conv2d_1/bias:@#<
:
_user_specified_name" Adam/m/cat_block_1/conv2d_1/bias:B">
<
_user_specified_name$"Adam/v/cat_block_1/conv2d_1/kernel:B!>
<
_user_specified_name$"Adam/m/cat_block_1/conv2d_1/kernel:I E
C
_user_specified_name+)Adam/v/cat_block/batch_normalization/beta:IE
C
_user_specified_name+)Adam/m/cat_block/batch_normalization/beta:JF
D
_user_specified_name,*Adam/v/cat_block/batch_normalization/gamma:JF
D
_user_specified_name,*Adam/m/cat_block/batch_normalization/gamma:<8
6
_user_specified_nameAdam/v/cat_block/conv2d/bias:<8
6
_user_specified_nameAdam/m/cat_block/conv2d/bias:>:
8
_user_specified_name Adam/v/cat_block/conv2d/kernel:>:
8
_user_specified_name Adam/m/cat_block/conv2d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:QM
K
_user_specified_name31cat_block_2/batch_normalization_2/moving_variance:MI
G
_user_specified_name/-cat_block_2/batch_normalization_2/moving_mean:FB
@
_user_specified_name(&cat_block_2/batch_normalization_2/beta:GC
A
_user_specified_name)'cat_block_2/batch_normalization_2/gamma:95
3
_user_specified_namecat_block_2/conv2d_2/bias:;7
5
_user_specified_namecat_block_2/conv2d_2/kernel:QM
K
_user_specified_name31cat_block_1/batch_normalization_1/moving_variance:MI
G
_user_specified_name/-cat_block_1/batch_normalization_1/moving_mean:F
B
@
_user_specified_name(&cat_block_1/batch_normalization_1/beta:G	C
A
_user_specified_name)'cat_block_1/batch_normalization_1/gamma:95
3
_user_specified_namecat_block_1/conv2d_1/bias:;7
5
_user_specified_namecat_block_1/conv2d_1/kernel:MI
G
_user_specified_name/-cat_block/batch_normalization/moving_variance:IE
C
_user_specified_name+)cat_block/batch_normalization/moving_mean:B>
<
_user_specified_name$"cat_block/batch_normalization/beta:C?
=
_user_specified_name%#cat_block/batch_normalization/gamma:51
/
_user_specified_namecat_block/conv2d/bias:73
1
_user_specified_namecat_block/conv2d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
&__inference_dense_layer_call_fn_107644

inputs
unknown:���
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_106827p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107640:&"
 
_user_specified_name107638:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
,__inference_cat_block_2_layer_call_fn_107555	
input#
unknown:��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_2_layer_call_and_return_conditional_losses_106796x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������

�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107551:&"
 
_user_specified_name107549:&"
 
_user_specified_name107547:&"
 
_user_specified_name107545:&"
 
_user_specified_name107543:&"
 
_user_specified_name107541:W S
0
_output_shapes
:����������

_user_specified_nameinput
�
J
.__inference_max_pooling2d_layer_call_fn_107769

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_106541�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
� 
�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107538	
inputB
'conv2d_1_conv2d_readvariableop_resource:@�7
(conv2d_1_biasadd_readvariableop_resource:	�<
-batch_normalization_1_readvariableop_resource:	�>
/batch_normalization_1_readvariableop_1_resource:	�M
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:	�
identity��5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_1/Conv2DConv2Dinput&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������//�k
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������//��
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:���������//�:�:�:�:�:*
epsilon%o�:*
is_training( �
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_1/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������11@: : : : : : 2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:V R
/
_output_shapes
:���������11@

_user_specified_nameinput
��
�*
"__inference__traced_restore_108489
file_prefixB
(assignvariableop_cat_block_conv2d_kernel:@6
(assignvariableop_1_cat_block_conv2d_bias:@D
6assignvariableop_2_cat_block_batch_normalization_gamma:@C
5assignvariableop_3_cat_block_batch_normalization_beta:@J
<assignvariableop_4_cat_block_batch_normalization_moving_mean:@N
@assignvariableop_5_cat_block_batch_normalization_moving_variance:@I
.assignvariableop_6_cat_block_1_conv2d_1_kernel:@�;
,assignvariableop_7_cat_block_1_conv2d_1_bias:	�I
:assignvariableop_8_cat_block_1_batch_normalization_1_gamma:	�H
9assignvariableop_9_cat_block_1_batch_normalization_1_beta:	�P
Aassignvariableop_10_cat_block_1_batch_normalization_1_moving_mean:	�T
Eassignvariableop_11_cat_block_1_batch_normalization_1_moving_variance:	�K
/assignvariableop_12_cat_block_2_conv2d_2_kernel:��<
-assignvariableop_13_cat_block_2_conv2d_2_bias:	�J
;assignvariableop_14_cat_block_2_batch_normalization_2_gamma:	�I
:assignvariableop_15_cat_block_2_batch_normalization_2_beta:	�P
Aassignvariableop_16_cat_block_2_batch_normalization_2_moving_mean:	�T
Eassignvariableop_17_cat_block_2_batch_normalization_2_moving_variance:	�5
 assignvariableop_18_dense_kernel:���-
assignvariableop_19_dense_bias:	�5
"assignvariableop_20_dense_1_kernel:	�.
 assignvariableop_21_dense_1_bias:'
assignvariableop_22_iteration:	 +
!assignvariableop_23_learning_rate: L
2assignvariableop_24_adam_m_cat_block_conv2d_kernel:@L
2assignvariableop_25_adam_v_cat_block_conv2d_kernel:@>
0assignvariableop_26_adam_m_cat_block_conv2d_bias:@>
0assignvariableop_27_adam_v_cat_block_conv2d_bias:@L
>assignvariableop_28_adam_m_cat_block_batch_normalization_gamma:@L
>assignvariableop_29_adam_v_cat_block_batch_normalization_gamma:@K
=assignvariableop_30_adam_m_cat_block_batch_normalization_beta:@K
=assignvariableop_31_adam_v_cat_block_batch_normalization_beta:@Q
6assignvariableop_32_adam_m_cat_block_1_conv2d_1_kernel:@�Q
6assignvariableop_33_adam_v_cat_block_1_conv2d_1_kernel:@�C
4assignvariableop_34_adam_m_cat_block_1_conv2d_1_bias:	�C
4assignvariableop_35_adam_v_cat_block_1_conv2d_1_bias:	�Q
Bassignvariableop_36_adam_m_cat_block_1_batch_normalization_1_gamma:	�Q
Bassignvariableop_37_adam_v_cat_block_1_batch_normalization_1_gamma:	�P
Aassignvariableop_38_adam_m_cat_block_1_batch_normalization_1_beta:	�P
Aassignvariableop_39_adam_v_cat_block_1_batch_normalization_1_beta:	�R
6assignvariableop_40_adam_m_cat_block_2_conv2d_2_kernel:��R
6assignvariableop_41_adam_v_cat_block_2_conv2d_2_kernel:��C
4assignvariableop_42_adam_m_cat_block_2_conv2d_2_bias:	�C
4assignvariableop_43_adam_v_cat_block_2_conv2d_2_bias:	�Q
Bassignvariableop_44_adam_m_cat_block_2_batch_normalization_2_gamma:	�Q
Bassignvariableop_45_adam_v_cat_block_2_batch_normalization_2_gamma:	�P
Aassignvariableop_46_adam_m_cat_block_2_batch_normalization_2_beta:	�P
Aassignvariableop_47_adam_v_cat_block_2_batch_normalization_2_beta:	�<
'assignvariableop_48_adam_m_dense_kernel:���<
'assignvariableop_49_adam_v_dense_kernel:���4
%assignvariableop_50_adam_m_dense_bias:	�4
%assignvariableop_51_adam_v_dense_bias:	�<
)assignvariableop_52_adam_m_dense_1_kernel:	�<
)assignvariableop_53_adam_v_dense_1_kernel:	�5
'assignvariableop_54_adam_m_dense_1_bias:5
'assignvariableop_55_adam_v_dense_1_bias:%
assignvariableop_56_total_1: %
assignvariableop_57_count_1: #
assignvariableop_58_total: #
assignvariableop_59_count: 
identity_61��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp(assignvariableop_cat_block_conv2d_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp(assignvariableop_1_cat_block_conv2d_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp6assignvariableop_2_cat_block_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp5assignvariableop_3_cat_block_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp<assignvariableop_4_cat_block_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp@assignvariableop_5_cat_block_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp.assignvariableop_6_cat_block_1_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp,assignvariableop_7_cat_block_1_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp:assignvariableop_8_cat_block_1_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp9assignvariableop_9_cat_block_1_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpAassignvariableop_10_cat_block_1_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpEassignvariableop_11_cat_block_1_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_cat_block_2_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp-assignvariableop_13_cat_block_2_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp;assignvariableop_14_cat_block_2_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp:assignvariableop_15_cat_block_2_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpAassignvariableop_16_cat_block_2_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpEassignvariableop_17_cat_block_2_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_iterationIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_m_cat_block_conv2d_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_v_cat_block_conv2d_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp0assignvariableop_26_adam_m_cat_block_conv2d_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_adam_v_cat_block_conv2d_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_m_cat_block_batch_normalization_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp>assignvariableop_29_adam_v_cat_block_batch_normalization_gammaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp=assignvariableop_30_adam_m_cat_block_batch_normalization_betaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp=assignvariableop_31_adam_v_cat_block_batch_normalization_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp6assignvariableop_32_adam_m_cat_block_1_conv2d_1_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_v_cat_block_1_conv2d_1_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_m_cat_block_1_conv2d_1_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_v_cat_block_1_conv2d_1_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOpBassignvariableop_36_adam_m_cat_block_1_batch_normalization_1_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpBassignvariableop_37_adam_v_cat_block_1_batch_normalization_1_gammaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpAassignvariableop_38_adam_m_cat_block_1_batch_normalization_1_betaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOpAassignvariableop_39_adam_v_cat_block_1_batch_normalization_1_betaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_m_cat_block_2_conv2d_2_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_v_cat_block_2_conv2d_2_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_m_cat_block_2_conv2d_2_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_v_cat_block_2_conv2d_2_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpBassignvariableop_44_adam_m_cat_block_2_batch_normalization_2_gammaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpBassignvariableop_45_adam_v_cat_block_2_batch_normalization_2_gammaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOpAassignvariableop_46_adam_m_cat_block_2_batch_normalization_2_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpAassignvariableop_47_adam_v_cat_block_2_batch_normalization_2_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_m_dense_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_v_dense_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp%assignvariableop_50_adam_m_dense_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp%assignvariableop_51_adam_v_dense_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_m_dense_1_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_v_dense_1_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_m_dense_1_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_v_dense_1_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_1Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_1Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpassignvariableop_58_totalIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_countIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%<!

_user_specified_namecount:%;!

_user_specified_nametotal:':#
!
_user_specified_name	count_1:'9#
!
_user_specified_name	total_1:38/
-
_user_specified_nameAdam/v/dense_1/bias:37/
-
_user_specified_nameAdam/m/dense_1/bias:561
/
_user_specified_nameAdam/v/dense_1/kernel:551
/
_user_specified_nameAdam/m/dense_1/kernel:14-
+
_user_specified_nameAdam/v/dense/bias:13-
+
_user_specified_nameAdam/m/dense/bias:32/
-
_user_specified_nameAdam/v/dense/kernel:31/
-
_user_specified_nameAdam/m/dense/kernel:M0I
G
_user_specified_name/-Adam/v/cat_block_2/batch_normalization_2/beta:M/I
G
_user_specified_name/-Adam/m/cat_block_2/batch_normalization_2/beta:N.J
H
_user_specified_name0.Adam/v/cat_block_2/batch_normalization_2/gamma:N-J
H
_user_specified_name0.Adam/m/cat_block_2/batch_normalization_2/gamma:@,<
:
_user_specified_name" Adam/v/cat_block_2/conv2d_2/bias:@+<
:
_user_specified_name" Adam/m/cat_block_2/conv2d_2/bias:B*>
<
_user_specified_name$"Adam/v/cat_block_2/conv2d_2/kernel:B)>
<
_user_specified_name$"Adam/m/cat_block_2/conv2d_2/kernel:M(I
G
_user_specified_name/-Adam/v/cat_block_1/batch_normalization_1/beta:M'I
G
_user_specified_name/-Adam/m/cat_block_1/batch_normalization_1/beta:N&J
H
_user_specified_name0.Adam/v/cat_block_1/batch_normalization_1/gamma:N%J
H
_user_specified_name0.Adam/m/cat_block_1/batch_normalization_1/gamma:@$<
:
_user_specified_name" Adam/v/cat_block_1/conv2d_1/bias:@#<
:
_user_specified_name" Adam/m/cat_block_1/conv2d_1/bias:B">
<
_user_specified_name$"Adam/v/cat_block_1/conv2d_1/kernel:B!>
<
_user_specified_name$"Adam/m/cat_block_1/conv2d_1/kernel:I E
C
_user_specified_name+)Adam/v/cat_block/batch_normalization/beta:IE
C
_user_specified_name+)Adam/m/cat_block/batch_normalization/beta:JF
D
_user_specified_name,*Adam/v/cat_block/batch_normalization/gamma:JF
D
_user_specified_name,*Adam/m/cat_block/batch_normalization/gamma:<8
6
_user_specified_nameAdam/v/cat_block/conv2d/bias:<8
6
_user_specified_nameAdam/m/cat_block/conv2d/bias:>:
8
_user_specified_name Adam/v/cat_block/conv2d/kernel:>:
8
_user_specified_name Adam/m/cat_block/conv2d/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:QM
K
_user_specified_name31cat_block_2/batch_normalization_2/moving_variance:MI
G
_user_specified_name/-cat_block_2/batch_normalization_2/moving_mean:FB
@
_user_specified_name(&cat_block_2/batch_normalization_2/beta:GC
A
_user_specified_name)'cat_block_2/batch_normalization_2/gamma:95
3
_user_specified_namecat_block_2/conv2d_2/bias:;7
5
_user_specified_namecat_block_2/conv2d_2/kernel:QM
K
_user_specified_name31cat_block_1/batch_normalization_1/moving_variance:MI
G
_user_specified_name/-cat_block_1/batch_normalization_1/moving_mean:F
B
@
_user_specified_name(&cat_block_1/batch_normalization_1/beta:G	C
A
_user_specified_name)'cat_block_1/batch_normalization_1/gamma:95
3
_user_specified_namecat_block_1/conv2d_1/bias:;7
5
_user_specified_namecat_block_1/conv2d_1/kernel:MI
G
_user_specified_name/-cat_block/batch_normalization/moving_variance:IE
C
_user_specified_name+)cat_block/batch_normalization/moving_mean:B>
<
_user_specified_name$"cat_block/batch_normalization/beta:C?
=
_user_specified_name%#cat_block/batch_normalization/gamma:51
/
_user_specified_namecat_block/conv2d/bias:73
1
_user_specified_namecat_block/conv2d/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
(__inference_cat_net_layer_call_fn_107099
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:���

unknown_18:	�

unknown_19:	�

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cat_net_layer_call_and_return_conditional_losses_107001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107095:&"
 
_user_specified_name107093:&"
 
_user_specified_name107091:&"
 
_user_specified_name107089:&"
 
_user_specified_name107087:&"
 
_user_specified_name107085:&"
 
_user_specified_name107083:&"
 
_user_specified_name107081:&"
 
_user_specified_name107079:&"
 
_user_specified_name107077:&"
 
_user_specified_name107075:&"
 
_user_specified_name107073:&
"
 
_user_specified_name107071:&	"
 
_user_specified_name107069:&"
 
_user_specified_name107067:&"
 
_user_specified_name107065:&"
 
_user_specified_name107063:&"
 
_user_specified_name107061:&"
 
_user_specified_name107059:&"
 
_user_specified_name107057:&"
 
_user_specified_name107055:&"
 
_user_specified_name107053:X T
/
_output_shapes
:���������dd
!
_user_specified_name	input_1
�
W
#__inference__update_step_xla_107291
gradient"
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:@: *
	_noinline(:($
"
_user_specified_name
variable:P L
&
_output_shapes
:@
"
_user_specified_name
gradient
�

�
4__inference_batch_normalization_layer_call_fn_107728

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_106510�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107724:&"
 
_user_specified_name107722:&"
 
_user_specified_name107720:&"
 
_user_specified_name107718:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107846

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
L
#__inference__update_step_xla_107341
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�/
�
C__inference_cat_net_layer_call_and_return_conditional_losses_106863
input_1*
cat_block_106719:@
cat_block_106721:@
cat_block_106723:@
cat_block_106725:@
cat_block_106727:@
cat_block_106729:@-
cat_block_1_106758:@�!
cat_block_1_106760:	�!
cat_block_1_106762:	�!
cat_block_1_106764:	�!
cat_block_1_106766:	�!
cat_block_1_106768:	�.
cat_block_2_106797:��!
cat_block_2_106799:	�!
cat_block_2_106801:	�!
cat_block_2_106803:	�!
cat_block_2_106805:	�!
cat_block_2_106807:	�!
dense_106828:���
dense_106830:	�!
dense_1_106857:	�
dense_1_106859:
identity��!cat_block/StatefulPartitionedCall�#cat_block_1/StatefulPartitionedCall�#cat_block_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dropout/StatefulPartitionedCall�
!cat_block/StatefulPartitionedCallStatefulPartitionedCallinput_1cat_block_106719cat_block_106721cat_block_106723cat_block_106725cat_block_106727cat_block_106729*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������11@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_cat_block_layer_call_and_return_conditional_losses_106718�
#cat_block_1/StatefulPartitionedCallStatefulPartitionedCall*cat_block/StatefulPartitionedCall:output:0cat_block_1_106758cat_block_1_106760cat_block_1_106762cat_block_1_106764cat_block_1_106766cat_block_1_106768*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_1_layer_call_and_return_conditional_losses_106757�
#cat_block_2/StatefulPartitionedCallStatefulPartitionedCall,cat_block_1/StatefulPartitionedCall:output:0cat_block_2_106797cat_block_2_106799cat_block_2_106801cat_block_2_106803cat_block_2_106805cat_block_2_106807*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������

�*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_cat_block_2_layer_call_and_return_conditional_losses_106796�
flatten/PartitionedCallPartitionedCall,cat_block_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_106815�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_106828dense_106830*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_106827�
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_106844�
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_106857dense_1_106859*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_106856w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^cat_block/StatefulPartitionedCall$^cat_block_1/StatefulPartitionedCall$^cat_block_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������dd: : : : : : : : : : : : : : : : : : : : : : 2F
!cat_block/StatefulPartitionedCall!cat_block/StatefulPartitionedCall2J
#cat_block_1/StatefulPartitionedCall#cat_block_1/StatefulPartitionedCall2J
#cat_block_2/StatefulPartitionedCall#cat_block_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:&"
 
_user_specified_name106859:&"
 
_user_specified_name106857:&"
 
_user_specified_name106830:&"
 
_user_specified_name106828:&"
 
_user_specified_name106807:&"
 
_user_specified_name106805:&"
 
_user_specified_name106803:&"
 
_user_specified_name106801:&"
 
_user_specified_name106799:&"
 
_user_specified_name106797:&"
 
_user_specified_name106768:&"
 
_user_specified_name106766:&
"
 
_user_specified_name106764:&	"
 
_user_specified_name106762:&"
 
_user_specified_name106760:&"
 
_user_specified_name106758:&"
 
_user_specified_name106729:&"
 
_user_specified_name106727:&"
 
_user_specified_name106725:&"
 
_user_specified_name106723:&"
 
_user_specified_name106721:&"
 
_user_specified_name106719:X T
/
_output_shapes
:���������dd
!
_user_specified_name	input_1
�
X
#__inference__update_step_xla_107311
gradient#
variable:@�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*(
_input_shapes
:@�: *
	_noinline(:($
"
_user_specified_name
variable:Q M
'
_output_shapes
:@�
"
_user_specified_name
gradient
�
a
C__inference_dropout_layer_call_and_return_conditional_losses_107682

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107908

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_2_layer_call_fn_107872

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_106654�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107868:&"
 
_user_specified_name107866:&"
 
_user_specified_name107864:&"
 
_user_specified_name107862:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
C__inference_dense_1_layer_call_and_return_conditional_losses_106856

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
*__inference_cat_block_layer_call_fn_107400	
input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������11@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_cat_block_layer_call_and_return_conditional_losses_106891w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������11@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������dd: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107396:&"
 
_user_specified_name107394:&"
 
_user_specified_name107392:&"
 
_user_specified_name107390:&"
 
_user_specified_name107388:&"
 
_user_specified_name107386:V R
/
_output_shapes
:���������dd

_user_specified_nameinput
�
Y
#__inference__update_step_xla_107331
gradient$
variable:��*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:��: *
	_noinline(:($
"
_user_specified_name
variable:R N
(
_output_shapes
:��
"
_user_specified_name
gradient
�

�
*__inference_cat_block_layer_call_fn_107383	
input!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������11@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_cat_block_layer_call_and_return_conditional_losses_106718w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������11@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������dd: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107379:&"
 
_user_specified_name107377:&"
 
_user_specified_name107375:&"
 
_user_specified_name107373:&"
 
_user_specified_name107371:&"
 
_user_specified_name107369:V R
/
_output_shapes
:���������dd

_user_specified_nameinput
�
L
0__inference_max_pooling2d_1_layer_call_fn_107841

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_106613�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_106510

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_107774

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107918

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107890

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�

�
6__inference_batch_normalization_1_layer_call_fn_107787

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,����������������������������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_106564�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,����������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107783:&"
 
_user_specified_name107781:&"
 
_user_specified_name107779:&"
 
_user_specified_name107777:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs
�
�
(__inference_cat_net_layer_call_fn_107050
input_1!
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@$
	unknown_5:@�
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�&

unknown_11:��

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�

unknown_17:���

unknown_18:	�

unknown_19:	�

unknown_20:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_cat_net_layer_call_and_return_conditional_losses_106863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������dd: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name107046:&"
 
_user_specified_name107044:&"
 
_user_specified_name107042:&"
 
_user_specified_name107040:&"
 
_user_specified_name107038:&"
 
_user_specified_name107036:&"
 
_user_specified_name107034:&"
 
_user_specified_name107032:&"
 
_user_specified_name107030:&"
 
_user_specified_name107028:&"
 
_user_specified_name107026:&"
 
_user_specified_name107024:&
"
 
_user_specified_name107022:&	"
 
_user_specified_name107020:&"
 
_user_specified_name107018:&"
 
_user_specified_name107016:&"
 
_user_specified_name107014:&"
 
_user_specified_name107012:&"
 
_user_specified_name107010:&"
 
_user_specified_name107008:&"
 
_user_specified_name107006:&"
 
_user_specified_name107004:X T
/
_output_shapes
:���������dd
!
_user_specified_name	input_1
�
L
#__inference__update_step_xla_107346
gradient
variable:	�*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:�: *
	_noinline(:($
"
_user_specified_name
variable:E A

_output_shapes	
:�
"
_user_specified_name
gradient
�(
�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_106796	
inputC
'conv2d_2_conv2d_readvariableop_resource:��7
(conv2d_2_biasadd_readvariableop_resource:	�<
-batch_normalization_2_readvariableop_resource:	�>
/batch_normalization_2_readvariableop_1_resource:	�M
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:	�O
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:	�
identity��$batch_normalization_2/AssignNewValue�&batch_normalization_2/AssignNewValue_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_2/Conv2DConv2Dinput&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingVALID*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:����������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_2/FusedBatchNormV3:y:0*0
_output_shapes
:���������

�*
ksize
*
paddingVALID*
strides
x
IdentityIdentity max_pooling2d_2/MaxPool:output:0^NoOp*
T0*0
_output_shapes
:���������

��
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:����������: : : : : : 2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
0
_output_shapes
:����������

_user_specified_nameinput
�
R
#__inference__update_step_xla_107351
gradient
variable:���*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*)
_input_shapes
:����������: *
	_noinline(:($
"
_user_specified_name
variable:R N
(
_output_shapes
:����������
"
_user_specified_name
gradient
�
�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_106564

inputs&
readvariableop_resource:	�(
readvariableop_1_resource:	�7
(fusedbatchnormv3_readvariableop_resource:	�9
*fusedbatchnormv3_readvariableop_1_resource:	�
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:�*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,����������������������������:�:�:�:�:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,�����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,����������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:j f
B
_output_shapes0
.:,����������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������dd<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
cnn1
	cnn2

cnn3
flatten
	connected
dropout

classifier
	optimizer

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
 15
!16
"17
#18
$19
%20
&21"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
 11
#12
$13
%14
&15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
'non_trainable_variables

(layers
)metrics
*layer_regularization_losses
+layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
,trace_0
-trace_12�
(__inference_cat_net_layer_call_fn_107050
(__inference_cat_net_layer_call_fn_107099�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z,trace_0z-trace_1
�
.trace_0
/trace_12�
C__inference_cat_net_layer_call_and_return_conditional_losses_106863
C__inference_cat_net_layer_call_and_return_conditional_losses_107001�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z.trace_0z/trace_1
�B�
!__inference__wrapped_model_106474input_1"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6conv
7bn
8pooling"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?conv
@bn
Apooling"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Hconv
Ibn
Jpooling"
_tf_keras_layer
�
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]_random_generator"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
�
d
_variables
e_iterations
f_learning_rate
g_index_dict
h
_momentums
i_velocities
j_update_step_xla"
experimentalOptimizer
,
kserving_default"
signature_map
1:/@2cat_block/conv2d/kernel
#:!@2cat_block/conv2d/bias
1:/@2#cat_block/batch_normalization/gamma
0:.@2"cat_block/batch_normalization/beta
9:7@ (2)cat_block/batch_normalization/moving_mean
=:;@ (2-cat_block/batch_normalization/moving_variance
6:4@�2cat_block_1/conv2d_1/kernel
(:&�2cat_block_1/conv2d_1/bias
6:4�2'cat_block_1/batch_normalization_1/gamma
5:3�2&cat_block_1/batch_normalization_1/beta
>:<� (2-cat_block_1/batch_normalization_1/moving_mean
B:@� (21cat_block_1/batch_normalization_1/moving_variance
7:5��2cat_block_2/conv2d_2/kernel
(:&�2cat_block_2/conv2d_2/bias
6:4�2'cat_block_2/batch_normalization_2/gamma
5:3�2&cat_block_2/batch_normalization_2/beta
>:<� (2-cat_block_2/batch_normalization_2/moving_mean
B:@� (21cat_block_2/batch_normalization_2/moving_variance
!:���2dense/kernel
:�2
dense/bias
!:	�2dense_1/kernel
:2dense_1/bias
J
0
1
2
3
!4
"5"
trackable_list_wrapper
Q
0
	1

2
3
4
5
6"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_cat_net_layer_call_fn_107050input_1"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_cat_net_layer_call_fn_107099input_1"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_cat_net_layer_call_and_return_conditional_losses_106863input_1"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_cat_net_layer_call_and_return_conditional_losses_107001input_1"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�
strace_0
ttrace_12�
*__inference_cat_block_layer_call_fn_107383
*__inference_cat_block_layer_call_fn_107400�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zstrace_0zttrace_1
�
utrace_0
vtrace_12�
E__inference_cat_block_layer_call_and_return_conditional_losses_107426
E__inference_cat_block_layer_call_and_return_conditional_losses_107452�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zutrace_0zvtrace_1
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

kernel
bias
 }_jit_compiled_convolution_op"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_cat_block_1_layer_call_fn_107469
,__inference_cat_block_1_layer_call_fn_107486�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107512
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107538�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
J
0
1
2
 3
!4
"5"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
,__inference_cat_block_2_layer_call_fn_107555
,__inference_cat_block_2_layer_call_fn_107572�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107598
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107624�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

kernel
bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis
	gamma
 beta
!moving_mean
"moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_flatten_layer_call_fn_107629�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_flatten_layer_call_and_return_conditional_losses_107635�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_dense_layer_call_fn_107644�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_dense_layer_call_and_return_conditional_losses_107655�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_dropout_layer_call_fn_107660
(__inference_dropout_layer_call_fn_107665�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_dropout_layer_call_and_return_conditional_losses_107677
C__inference_dropout_layer_call_and_return_conditional_losses_107682�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_1_layer_call_fn_107691�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_1_layer_call_and_return_conditional_losses_107702�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
e0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�	
�trace_0
�trace_1
�trace_2
�trace_3
�trace_4
�trace_5
�trace_6
�trace_7
�trace_8
�trace_9
�trace_10
�trace_11
�trace_12
�trace_13
�trace_14
�trace_152�
#__inference__update_step_xla_107291
#__inference__update_step_xla_107296
#__inference__update_step_xla_107301
#__inference__update_step_xla_107306
#__inference__update_step_xla_107311
#__inference__update_step_xla_107316
#__inference__update_step_xla_107321
#__inference__update_step_xla_107326
#__inference__update_step_xla_107331
#__inference__update_step_xla_107336
#__inference__update_step_xla_107341
#__inference__update_step_xla_107346
#__inference__update_step_xla_107351
#__inference__update_step_xla_107356
#__inference__update_step_xla_107361
#__inference__update_step_xla_107366�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0z�trace_0z�trace_1z�trace_2z�trace_3z�trace_4z�trace_5z�trace_6z�trace_7z�trace_8z�trace_9z�trace_10z�trace_11z�trace_12z�trace_13z�trace_14z�trace_15
�B�
$__inference_signature_wrapper_107286input_1"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�
	jinput_1
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
.
0
1"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_cat_block_layer_call_fn_107383input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_cat_block_layer_call_fn_107400input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_cat_block_layer_call_and_return_conditional_losses_107426input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_cat_block_layer_call_and_return_conditional_losses_107452input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
4__inference_batch_normalization_layer_call_fn_107715
4__inference_batch_normalization_layer_call_fn_107728�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107746
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107764�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_max_pooling2d_layer_call_fn_107769�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_107774�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
0
1"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_cat_block_1_layer_call_fn_107469input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_cat_block_1_layer_call_fn_107486input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107512input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107538input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_1_layer_call_fn_107787
6__inference_batch_normalization_1_layer_call_fn_107800�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107818
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107836�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_1_layer_call_fn_107841�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107846�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
!0
"1"
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_cat_block_2_layer_call_fn_107555input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_cat_block_2_layer_call_fn_107572input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107598input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107624input"�
���
FullArgSpec 
args�
jinput

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
0
 1
!2
"3"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_batch_normalization_2_layer_call_fn_107859
6__inference_batch_normalization_2_layer_call_fn_107872�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107890
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107908�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_max_pooling2d_2_layer_call_fn_107913�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107918�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
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
�B�
(__inference_flatten_layer_call_fn_107629inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_flatten_layer_call_and_return_conditional_losses_107635inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
&__inference_dense_layer_call_fn_107644inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dense_layer_call_and_return_conditional_losses_107655inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dropout_layer_call_fn_107660inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_dropout_layer_call_fn_107665inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_107677inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dropout_layer_call_and_return_conditional_losses_107682inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
(__inference_dense_1_layer_call_fn_107691inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_1_layer_call_and_return_conditional_losses_107702inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
6:4@2Adam/m/cat_block/conv2d/kernel
6:4@2Adam/v/cat_block/conv2d/kernel
(:&@2Adam/m/cat_block/conv2d/bias
(:&@2Adam/v/cat_block/conv2d/bias
6:4@2*Adam/m/cat_block/batch_normalization/gamma
6:4@2*Adam/v/cat_block/batch_normalization/gamma
5:3@2)Adam/m/cat_block/batch_normalization/beta
5:3@2)Adam/v/cat_block/batch_normalization/beta
;:9@�2"Adam/m/cat_block_1/conv2d_1/kernel
;:9@�2"Adam/v/cat_block_1/conv2d_1/kernel
-:+�2 Adam/m/cat_block_1/conv2d_1/bias
-:+�2 Adam/v/cat_block_1/conv2d_1/bias
;:9�2.Adam/m/cat_block_1/batch_normalization_1/gamma
;:9�2.Adam/v/cat_block_1/batch_normalization_1/gamma
::8�2-Adam/m/cat_block_1/batch_normalization_1/beta
::8�2-Adam/v/cat_block_1/batch_normalization_1/beta
<::��2"Adam/m/cat_block_2/conv2d_2/kernel
<::��2"Adam/v/cat_block_2/conv2d_2/kernel
-:+�2 Adam/m/cat_block_2/conv2d_2/bias
-:+�2 Adam/v/cat_block_2/conv2d_2/bias
;:9�2.Adam/m/cat_block_2/batch_normalization_2/gamma
;:9�2.Adam/v/cat_block_2/batch_normalization_2/gamma
::8�2-Adam/m/cat_block_2/batch_normalization_2/beta
::8�2-Adam/v/cat_block_2/batch_normalization_2/beta
&:$���2Adam/m/dense/kernel
&:$���2Adam/v/dense/kernel
:�2Adam/m/dense/bias
:�2Adam/v/dense/bias
&:$	�2Adam/m/dense_1/kernel
&:$	�2Adam/v/dense_1/kernel
:2Adam/m/dense_1/bias
:2Adam/v/dense_1/bias
�B�
#__inference__update_step_xla_107291gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107296gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107301gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107306gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107311gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107316gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107321gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107326gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107331gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107336gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107341gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107346gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107351gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107356gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107361gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference__update_step_xla_107366gradientvariable"�
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
4__inference_batch_normalization_layer_call_fn_107715inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
4__inference_batch_normalization_layer_call_fn_107728inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107746inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107764inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
.__inference_max_pooling2d_layer_call_fn_107769inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_107774inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_1_layer_call_fn_107787inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_1_layer_call_fn_107800inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107818inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107836inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_max_pooling2d_1_layer_call_fn_107841inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107846inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_batch_normalization_2_layer_call_fn_107859inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_batch_normalization_2_layer_call_fn_107872inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107890inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107908inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
0__inference_max_pooling2d_2_layer_call_fn_107913inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107918inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__update_step_xla_107291~x�u
n�k
!�
gradient@
<�9	%�"
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107296f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107301f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107306f`�]
V�S
�
gradient@
0�-	�
�@
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107311�z�w
p�m
"�
gradient@�
=�:	&�#
�@�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107316hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107321hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107326hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107331�|�y
r�o
#� 
gradient��
>�;	'�$
���
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107336hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107341hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107346hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107351{u�r
k�h
#� 
gradient����������
7�4	 �
����
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107356hb�_
X�U
�
gradient�
1�.	�
��
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107361pj�g
`�]
�
gradient	�
5�2	�
�	�
�
p
` VariableSpec 
`������?
� "
 �
#__inference__update_step_xla_107366f`�]
V�S
�
gradient
0�-	�
�
�
p
` VariableSpec 
`������?
� "
 �
!__inference__wrapped_model_106474� !"#$%&8�5
.�+
)�&
input_1���������dd
� "3�0
.
output_1"�
output_1����������
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107818�R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_107836�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_1_layer_call_fn_107787�R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_1_layer_call_fn_107800�R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107890� !"R�O
H�E
;�8
inputs,����������������������������
p

 
� "G�D
=�:
tensor_0,����������������������������
� �
Q__inference_batch_normalization_2_layer_call_and_return_conditional_losses_107908� !"R�O
H�E
;�8
inputs,����������������������������
p 

 
� "G�D
=�:
tensor_0,����������������������������
� �
6__inference_batch_normalization_2_layer_call_fn_107859� !"R�O
H�E
;�8
inputs,����������������������������
p

 
� "<�9
unknown,�����������������������������
6__inference_batch_normalization_2_layer_call_fn_107872� !"R�O
H�E
;�8
inputs,����������������������������
p 

 
� "<�9
unknown,�����������������������������
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107746�Q�N
G�D
:�7
inputs+���������������������������@
p

 
� "F�C
<�9
tensor_0+���������������������������@
� �
O__inference_batch_normalization_layer_call_and_return_conditional_losses_107764�Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� "F�C
<�9
tensor_0+���������������������������@
� �
4__inference_batch_normalization_layer_call_fn_107715�Q�N
G�D
:�7
inputs+���������������������������@
p

 
� ";�8
unknown+���������������������������@�
4__inference_batch_normalization_layer_call_fn_107728�Q�N
G�D
:�7
inputs+���������������������������@
p 

 
� ";�8
unknown+���������������������������@�
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107512{:�7
0�-
'�$
input���������11@
p
� "5�2
+�(
tensor_0����������
� �
G__inference_cat_block_1_layer_call_and_return_conditional_losses_107538{:�7
0�-
'�$
input���������11@
p 
� "5�2
+�(
tensor_0����������
� �
,__inference_cat_block_1_layer_call_fn_107469p:�7
0�-
'�$
input���������11@
p
� "*�'
unknown�����������
,__inference_cat_block_1_layer_call_fn_107486p:�7
0�-
'�$
input���������11@
p 
� "*�'
unknown�����������
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107598| !";�8
1�.
(�%
input����������
p
� "5�2
+�(
tensor_0���������

�
� �
G__inference_cat_block_2_layer_call_and_return_conditional_losses_107624| !";�8
1�.
(�%
input����������
p 
� "5�2
+�(
tensor_0���������

�
� �
,__inference_cat_block_2_layer_call_fn_107555q !";�8
1�.
(�%
input����������
p
� "*�'
unknown���������

��
,__inference_cat_block_2_layer_call_fn_107572q !";�8
1�.
(�%
input����������
p 
� "*�'
unknown���������

��
E__inference_cat_block_layer_call_and_return_conditional_losses_107426z:�7
0�-
'�$
input���������dd
p
� "4�1
*�'
tensor_0���������11@
� �
E__inference_cat_block_layer_call_and_return_conditional_losses_107452z:�7
0�-
'�$
input���������dd
p 
� "4�1
*�'
tensor_0���������11@
� �
*__inference_cat_block_layer_call_fn_107383o:�7
0�-
'�$
input���������dd
p
� ")�&
unknown���������11@�
*__inference_cat_block_layer_call_fn_107400o:�7
0�-
'�$
input���������dd
p 
� ")�&
unknown���������11@�
C__inference_cat_net_layer_call_and_return_conditional_losses_106863� !"#$%&<�9
2�/
)�&
input_1���������dd
p
� ",�)
"�
tensor_0���������
� �
C__inference_cat_net_layer_call_and_return_conditional_losses_107001� !"#$%&<�9
2�/
)�&
input_1���������dd
p 
� ",�)
"�
tensor_0���������
� �
(__inference_cat_net_layer_call_fn_107050y !"#$%&<�9
2�/
)�&
input_1���������dd
p
� "!�
unknown����������
(__inference_cat_net_layer_call_fn_107099y !"#$%&<�9
2�/
)�&
input_1���������dd
p 
� "!�
unknown����������
C__inference_dense_1_layer_call_and_return_conditional_losses_107702d%&0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
(__inference_dense_1_layer_call_fn_107691Y%&0�-
&�#
!�
inputs����������
� "!�
unknown����������
A__inference_dense_layer_call_and_return_conditional_losses_107655f#$1�.
'�$
"�
inputs�����������
� "-�*
#� 
tensor_0����������
� �
&__inference_dense_layer_call_fn_107644[#$1�.
'�$
"�
inputs�����������
� ""�
unknown�����������
C__inference_dropout_layer_call_and_return_conditional_losses_107677e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
C__inference_dropout_layer_call_and_return_conditional_losses_107682e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
(__inference_dropout_layer_call_fn_107660Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
(__inference_dropout_layer_call_fn_107665Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
C__inference_flatten_layer_call_and_return_conditional_losses_107635j8�5
.�+
)�&
inputs���������

�
� ".�+
$�!
tensor_0�����������
� �
(__inference_flatten_layer_call_fn_107629_8�5
.�+
)�&
inputs���������

�
� "#� 
unknown������������
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_107846�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_1_layer_call_fn_107841�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
K__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_107918�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
0__inference_max_pooling2d_2_layer_call_fn_107913�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_107774�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
.__inference_max_pooling2d_layer_call_fn_107769�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
$__inference_signature_wrapper_107286� !"#$%&C�@
� 
9�6
4
input_1)�&
input_1���������dd"3�0
.
output_1"�
output_1���������