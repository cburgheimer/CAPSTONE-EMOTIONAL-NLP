??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings* 
_output_shapes
:
??@*
dtype0
{
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*
shared_nameconv1d/kernel
t
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*#
_output_shapes
:@?*
dtype0
o
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d/bias
h
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes	
:?*
dtype0
?
Anticipation/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAnticipation/kernel
|
'Anticipation/kernel/Read/ReadVariableOpReadVariableOpAnticipation/kernel*
_output_shapes
:	?*
dtype0
z
Anticipation/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAnticipation/bias
s
%Anticipation/bias/Read/ReadVariableOpReadVariableOpAnticipation/bias*
_output_shapes
:*
dtype0
u
Anger/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameAnger/kernel
n
 Anger/kernel/Read/ReadVariableOpReadVariableOpAnger/kernel*
_output_shapes
:	?*
dtype0
l

Anger/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Anger/bias
e
Anger/bias/Read/ReadVariableOpReadVariableOp
Anger/bias*
_output_shapes
:*
dtype0
y
Disgust/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameDisgust/kernel
r
"Disgust/kernel/Read/ReadVariableOpReadVariableOpDisgust/kernel*
_output_shapes
:	?*
dtype0
p
Disgust/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDisgust/bias
i
 Disgust/bias/Read/ReadVariableOpReadVariableOpDisgust/bias*
_output_shapes
:*
dtype0
s
Fear/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameFear/kernel
l
Fear/kernel/Read/ReadVariableOpReadVariableOpFear/kernel*
_output_shapes
:	?*
dtype0
j
	Fear/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	Fear/bias
c
Fear/bias/Read/ReadVariableOpReadVariableOp	Fear/bias*
_output_shapes
:*
dtype0
q

Joy/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name
Joy/kernel
j
Joy/kernel/Read/ReadVariableOpReadVariableOp
Joy/kernel*
_output_shapes
:	?*
dtype0
h
Joy/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Joy/bias
a
Joy/bias/Read/ReadVariableOpReadVariableOpJoy/bias*
_output_shapes
:*
dtype0
y
Sadness/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameSadness/kernel
r
"Sadness/kernel/Read/ReadVariableOpReadVariableOpSadness/kernel*
_output_shapes
:	?*
dtype0
p
Sadness/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameSadness/bias
i
 Sadness/bias/Read/ReadVariableOpReadVariableOpSadness/bias*
_output_shapes
:*
dtype0
{
Surprise/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameSurprise/kernel
t
#Surprise/kernel/Read/ReadVariableOpReadVariableOpSurprise/kernel*
_output_shapes
:	?*
dtype0
r
Surprise/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameSurprise/bias
k
!Surprise/bias/Read/ReadVariableOpReadVariableOpSurprise/bias*
_output_shapes
:*
dtype0
u
Trust/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_nameTrust/kernel
n
 Trust/kernel/Read/ReadVariableOpReadVariableOpTrust/kernel*
_output_shapes
:	?*
dtype0
l

Trust/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
Trust/bias
e
Trust/bias/Read/ReadVariableOpReadVariableOp
Trust/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
j
Adamax/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdamax/iter
c
Adamax/iter/Read/ReadVariableOpReadVariableOpAdamax/iter*
_output_shapes
: *
dtype0	
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
d
total_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_10
]
total_10/Read/ReadVariableOpReadVariableOptotal_10*
_output_shapes
: *
dtype0
d
count_10VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_10
]
count_10/Read/ReadVariableOpReadVariableOpcount_10*
_output_shapes
: *
dtype0
d
total_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_11
]
total_11/Read/ReadVariableOpReadVariableOptotal_11*
_output_shapes
: *
dtype0
d
count_11VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_11
]
count_11/Read/ReadVariableOpReadVariableOpcount_11*
_output_shapes
: *
dtype0
d
total_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_12
]
total_12/Read/ReadVariableOpReadVariableOptotal_12*
_output_shapes
: *
dtype0
d
count_12VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_12
]
count_12/Read/ReadVariableOpReadVariableOpcount_12*
_output_shapes
: *
dtype0
d
total_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_13
]
total_13/Read/ReadVariableOpReadVariableOptotal_13*
_output_shapes
: *
dtype0
d
count_13VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_13
]
count_13/Read/ReadVariableOpReadVariableOpcount_13*
_output_shapes
: *
dtype0
d
total_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_14
]
total_14/Read/ReadVariableOpReadVariableOptotal_14*
_output_shapes
: *
dtype0
d
count_14VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_14
]
count_14/Read/ReadVariableOpReadVariableOpcount_14*
_output_shapes
: *
dtype0
d
total_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_15
]
total_15/Read/ReadVariableOpReadVariableOptotal_15*
_output_shapes
: *
dtype0
d
count_15VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_15
]
count_15/Read/ReadVariableOpReadVariableOpcount_15*
_output_shapes
: *
dtype0
d
total_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
total_16
]
total_16/Read/ReadVariableOpReadVariableOptotal_16*
_output_shapes
: *
dtype0
d
count_16VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
count_16
]
count_16/Read/ReadVariableOpReadVariableOpcount_16*
_output_shapes
: *
dtype0
?
Adamax/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*0
shared_name!Adamax/embedding_1/embeddings/m
?
3Adamax/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOpAdamax/embedding_1/embeddings/m* 
_output_shapes
:
??@*
dtype0
?
Adamax/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdamax/conv1d/kernel/m
?
*Adamax/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/conv1d/kernel/m*#
_output_shapes
:@?*
dtype0
?
Adamax/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdamax/conv1d/bias/m
z
(Adamax/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdamax/conv1d/bias/m*
_output_shapes	
:?*
dtype0
?
Adamax/Anticipation/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_nameAdamax/Anticipation/kernel/m
?
0Adamax/Anticipation/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Anticipation/kernel/m*
_output_shapes
:	?*
dtype0
?
Adamax/Anticipation/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdamax/Anticipation/bias/m
?
.Adamax/Anticipation/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Anticipation/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Anger/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdamax/Anger/kernel/m
?
)Adamax/Anger/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Anger/kernel/m*
_output_shapes
:	?*
dtype0
~
Adamax/Anger/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamax/Anger/bias/m
w
'Adamax/Anger/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Anger/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Disgust/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdamax/Disgust/kernel/m
?
+Adamax/Disgust/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Disgust/kernel/m*
_output_shapes
:	?*
dtype0
?
Adamax/Disgust/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/Disgust/bias/m
{
)Adamax/Disgust/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Disgust/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Fear/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdamax/Fear/kernel/m
~
(Adamax/Fear/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Fear/kernel/m*
_output_shapes
:	?*
dtype0
|
Adamax/Fear/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdamax/Fear/bias/m
u
&Adamax/Fear/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Fear/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Joy/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdamax/Joy/kernel/m
|
'Adamax/Joy/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Joy/kernel/m*
_output_shapes
:	?*
dtype0
z
Adamax/Joy/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdamax/Joy/bias/m
s
%Adamax/Joy/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Joy/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Sadness/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdamax/Sadness/kernel/m
?
+Adamax/Sadness/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Sadness/kernel/m*
_output_shapes
:	?*
dtype0
?
Adamax/Sadness/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/Sadness/bias/m
{
)Adamax/Sadness/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Sadness/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Surprise/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdamax/Surprise/kernel/m
?
,Adamax/Surprise/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Surprise/kernel/m*
_output_shapes
:	?*
dtype0
?
Adamax/Surprise/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/Surprise/bias/m
}
*Adamax/Surprise/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Surprise/bias/m*
_output_shapes
:*
dtype0
?
Adamax/Trust/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdamax/Trust/kernel/m
?
)Adamax/Trust/kernel/m/Read/ReadVariableOpReadVariableOpAdamax/Trust/kernel/m*
_output_shapes
:	?*
dtype0
~
Adamax/Trust/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamax/Trust/bias/m
w
'Adamax/Trust/bias/m/Read/ReadVariableOpReadVariableOpAdamax/Trust/bias/m*
_output_shapes
:*
dtype0
?
Adamax/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??@*0
shared_name!Adamax/embedding_1/embeddings/v
?
3Adamax/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOpAdamax/embedding_1/embeddings/v* 
_output_shapes
:
??@*
dtype0
?
Adamax/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*'
shared_nameAdamax/conv1d/kernel/v
?
*Adamax/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/conv1d/kernel/v*#
_output_shapes
:@?*
dtype0
?
Adamax/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdamax/conv1d/bias/v
z
(Adamax/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdamax/conv1d/bias/v*
_output_shapes	
:?*
dtype0
?
Adamax/Anticipation/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*-
shared_nameAdamax/Anticipation/kernel/v
?
0Adamax/Anticipation/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Anticipation/kernel/v*
_output_shapes
:	?*
dtype0
?
Adamax/Anticipation/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdamax/Anticipation/bias/v
?
.Adamax/Anticipation/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Anticipation/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Anger/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdamax/Anger/kernel/v
?
)Adamax/Anger/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Anger/kernel/v*
_output_shapes
:	?*
dtype0
~
Adamax/Anger/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamax/Anger/bias/v
w
'Adamax/Anger/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Anger/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Disgust/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdamax/Disgust/kernel/v
?
+Adamax/Disgust/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Disgust/kernel/v*
_output_shapes
:	?*
dtype0
?
Adamax/Disgust/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/Disgust/bias/v
{
)Adamax/Disgust/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Disgust/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Fear/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*%
shared_nameAdamax/Fear/kernel/v
~
(Adamax/Fear/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Fear/kernel/v*
_output_shapes
:	?*
dtype0
|
Adamax/Fear/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdamax/Fear/bias/v
u
&Adamax/Fear/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Fear/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Joy/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdamax/Joy/kernel/v
|
'Adamax/Joy/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Joy/kernel/v*
_output_shapes
:	?*
dtype0
z
Adamax/Joy/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdamax/Joy/bias/v
s
%Adamax/Joy/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Joy/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Sadness/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*(
shared_nameAdamax/Sadness/kernel/v
?
+Adamax/Sadness/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Sadness/kernel/v*
_output_shapes
:	?*
dtype0
?
Adamax/Sadness/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamax/Sadness/bias/v
{
)Adamax/Sadness/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Sadness/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Surprise/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdamax/Surprise/kernel/v
?
,Adamax/Surprise/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Surprise/kernel/v*
_output_shapes
:	?*
dtype0
?
Adamax/Surprise/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdamax/Surprise/bias/v
}
*Adamax/Surprise/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Surprise/bias/v*
_output_shapes
:*
dtype0
?
Adamax/Trust/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdamax/Trust/kernel/v
?
)Adamax/Trust/kernel/v/Read/ReadVariableOpReadVariableOpAdamax/Trust/kernel/v*
_output_shapes
:	?*
dtype0
~
Adamax/Trust/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamax/Trust/bias/v
w
'Adamax/Trust/bias/v/Read/ReadVariableOpReadVariableOpAdamax/Trust/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*΅
valueÅB?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
h

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
h

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
h

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?

Xbeta_1

Ybeta_2
	Zdecay
[learning_rate
\iterm?m?m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?v?v?v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?
 
?
0
1
2
(3
)4
.5
/6
47
58
:9
;10
@11
A12
F13
G14
L15
M16
R17
S18
?
0
1
2
(3
)4
.5
/6
47
58
:9
;10
@11
A12
F13
G14
L15
M16
R17
S18
 
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUEconv1d/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
 	variables
!trainable_variables
"regularization_losses
 
 
 
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
$	variables
%trainable_variables
&regularization_losses
_]
VARIABLE_VALUEAnticipation/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEAnticipation/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
*	variables
+trainable_variables
,regularization_losses
XV
VARIABLE_VALUEAnger/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Anger/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
ZX
VARIABLE_VALUEDisgust/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDisgust/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
WU
VARIABLE_VALUEFear/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	Fear/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
VT
VARIABLE_VALUE
Joy/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEJoy/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

@0
A1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
ZX
VARIABLE_VALUESadness/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUESadness/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
[Y
VARIABLE_VALUESurprise/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUESurprise/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
XV
VARIABLE_VALUETrust/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
Trust/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdamax/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
 
^
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
10
11
12
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
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

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_34keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_34keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_44keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_44keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_54keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_54keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_64keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_64keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_74keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_74keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_84keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_84keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_94keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_94keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_105keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_105keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_115keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_115keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_125keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_125keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_135keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_135keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_145keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_145keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_155keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_155keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
SQ
VARIABLE_VALUEtotal_165keras_api/metrics/16/total/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEcount_165keras_api/metrics/16/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUEAdamax/embedding_1/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv1d/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv1d/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdamax/Anticipation/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdamax/Anticipation/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/Anger/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/Anger/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/Disgust/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/Disgust/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/Fear/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamax/Fear/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/Joy/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamax/Joy/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/Sadness/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/Sadness/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdamax/Surprise/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/Surprise/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/Trust/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/Trust/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdamax/embedding_1/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamax/conv1d/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamax/conv1d/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdamax/Anticipation/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdamax/Anticipation/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/Anger/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/Anger/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/Disgust/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/Disgust/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/Fear/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamax/Fear/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/Joy/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamax/Joy/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamax/Sadness/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamax/Sadness/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdamax/Surprise/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamax/Surprise/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamax/Trust/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamax/Trust/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
serving_default_main_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_main_inputembedding_1/embeddingsconv1d/kernelconv1d/biasTrust/kernel
Trust/biasSurprise/kernelSurprise/biasSadness/kernelSadness/bias
Joy/kernelJoy/biasFear/kernel	Fear/biasDisgust/kernelDisgust/biasAnger/kernel
Anger/biasAnticipation/kernelAnticipation/bias*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_190931
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*embedding_1/embeddings/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp'Anticipation/kernel/Read/ReadVariableOp%Anticipation/bias/Read/ReadVariableOp Anger/kernel/Read/ReadVariableOpAnger/bias/Read/ReadVariableOp"Disgust/kernel/Read/ReadVariableOp Disgust/bias/Read/ReadVariableOpFear/kernel/Read/ReadVariableOpFear/bias/Read/ReadVariableOpJoy/kernel/Read/ReadVariableOpJoy/bias/Read/ReadVariableOp"Sadness/kernel/Read/ReadVariableOp Sadness/bias/Read/ReadVariableOp#Surprise/kernel/Read/ReadVariableOp!Surprise/bias/Read/ReadVariableOp Trust/kernel/Read/ReadVariableOpTrust/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdamax/iter/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_3/Read/ReadVariableOpcount_3/Read/ReadVariableOptotal_4/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_5/Read/ReadVariableOpcount_5/Read/ReadVariableOptotal_6/Read/ReadVariableOpcount_6/Read/ReadVariableOptotal_7/Read/ReadVariableOpcount_7/Read/ReadVariableOptotal_8/Read/ReadVariableOpcount_8/Read/ReadVariableOptotal_9/Read/ReadVariableOpcount_9/Read/ReadVariableOptotal_10/Read/ReadVariableOpcount_10/Read/ReadVariableOptotal_11/Read/ReadVariableOpcount_11/Read/ReadVariableOptotal_12/Read/ReadVariableOpcount_12/Read/ReadVariableOptotal_13/Read/ReadVariableOpcount_13/Read/ReadVariableOptotal_14/Read/ReadVariableOpcount_14/Read/ReadVariableOptotal_15/Read/ReadVariableOpcount_15/Read/ReadVariableOptotal_16/Read/ReadVariableOpcount_16/Read/ReadVariableOp3Adamax/embedding_1/embeddings/m/Read/ReadVariableOp*Adamax/conv1d/kernel/m/Read/ReadVariableOp(Adamax/conv1d/bias/m/Read/ReadVariableOp0Adamax/Anticipation/kernel/m/Read/ReadVariableOp.Adamax/Anticipation/bias/m/Read/ReadVariableOp)Adamax/Anger/kernel/m/Read/ReadVariableOp'Adamax/Anger/bias/m/Read/ReadVariableOp+Adamax/Disgust/kernel/m/Read/ReadVariableOp)Adamax/Disgust/bias/m/Read/ReadVariableOp(Adamax/Fear/kernel/m/Read/ReadVariableOp&Adamax/Fear/bias/m/Read/ReadVariableOp'Adamax/Joy/kernel/m/Read/ReadVariableOp%Adamax/Joy/bias/m/Read/ReadVariableOp+Adamax/Sadness/kernel/m/Read/ReadVariableOp)Adamax/Sadness/bias/m/Read/ReadVariableOp,Adamax/Surprise/kernel/m/Read/ReadVariableOp*Adamax/Surprise/bias/m/Read/ReadVariableOp)Adamax/Trust/kernel/m/Read/ReadVariableOp'Adamax/Trust/bias/m/Read/ReadVariableOp3Adamax/embedding_1/embeddings/v/Read/ReadVariableOp*Adamax/conv1d/kernel/v/Read/ReadVariableOp(Adamax/conv1d/bias/v/Read/ReadVariableOp0Adamax/Anticipation/kernel/v/Read/ReadVariableOp.Adamax/Anticipation/bias/v/Read/ReadVariableOp)Adamax/Anger/kernel/v/Read/ReadVariableOp'Adamax/Anger/bias/v/Read/ReadVariableOp+Adamax/Disgust/kernel/v/Read/ReadVariableOp)Adamax/Disgust/bias/v/Read/ReadVariableOp(Adamax/Fear/kernel/v/Read/ReadVariableOp&Adamax/Fear/bias/v/Read/ReadVariableOp'Adamax/Joy/kernel/v/Read/ReadVariableOp%Adamax/Joy/bias/v/Read/ReadVariableOp+Adamax/Sadness/kernel/v/Read/ReadVariableOp)Adamax/Sadness/bias/v/Read/ReadVariableOp,Adamax/Surprise/kernel/v/Read/ReadVariableOp*Adamax/Surprise/bias/v/Read/ReadVariableOp)Adamax/Trust/kernel/v/Read/ReadVariableOp'Adamax/Trust/bias/v/Read/ReadVariableOpConst*m
Tinf
d2b	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_191794
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsconv1d/kernelconv1d/biasAnticipation/kernelAnticipation/biasAnger/kernel
Anger/biasDisgust/kernelDisgust/biasFear/kernel	Fear/bias
Joy/kernelJoy/biasSadness/kernelSadness/biasSurprise/kernelSurprise/biasTrust/kernel
Trust/biasbeta_1beta_2decaylearning_rateAdamax/itertotalcounttotal_1count_1total_2count_2total_3count_3total_4count_4total_5count_5total_6count_6total_7count_7total_8count_8total_9count_9total_10count_10total_11count_11total_12count_12total_13count_13total_14count_14total_15count_15total_16count_16Adamax/embedding_1/embeddings/mAdamax/conv1d/kernel/mAdamax/conv1d/bias/mAdamax/Anticipation/kernel/mAdamax/Anticipation/bias/mAdamax/Anger/kernel/mAdamax/Anger/bias/mAdamax/Disgust/kernel/mAdamax/Disgust/bias/mAdamax/Fear/kernel/mAdamax/Fear/bias/mAdamax/Joy/kernel/mAdamax/Joy/bias/mAdamax/Sadness/kernel/mAdamax/Sadness/bias/mAdamax/Surprise/kernel/mAdamax/Surprise/bias/mAdamax/Trust/kernel/mAdamax/Trust/bias/mAdamax/embedding_1/embeddings/vAdamax/conv1d/kernel/vAdamax/conv1d/bias/vAdamax/Anticipation/kernel/vAdamax/Anticipation/bias/vAdamax/Anger/kernel/vAdamax/Anger/bias/vAdamax/Disgust/kernel/vAdamax/Disgust/bias/vAdamax/Fear/kernel/vAdamax/Fear/bias/vAdamax/Joy/kernel/vAdamax/Joy/bias/vAdamax/Sadness/kernel/vAdamax/Sadness/bias/vAdamax/Surprise/kernel/vAdamax/Surprise/bias/vAdamax/Trust/kernel/vAdamax/Trust/bias/v*l
Tine
c2a*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_192092??
?

?
C__inference_Disgust_layer_call_and_return_conditional_losses_190283

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
e
F__inference_dropout_39_layer_call_and_return_conditional_losses_190486

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
Q
5__inference_global_max_pooling1d_layer_call_fn_191277

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190178a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_Fear_layer_call_and_return_conditional_losses_190266

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_Anger_layer_call_fn_191345

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Anger_layer_call_and_return_conditional_losses_190300o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190178

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:??????????U
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?r
?
!__inference__wrapped_model_190118

main_input?
+model_1_embedding_1_embedding_lookup_190034:
??@Q
:model_1_conv1d_conv1d_expanddims_1_readvariableop_resource:@?=
.model_1_conv1d_biasadd_readvariableop_resource:	??
,model_1_trust_matmul_readvariableop_resource:	?;
-model_1_trust_biasadd_readvariableop_resource:B
/model_1_surprise_matmul_readvariableop_resource:	?>
0model_1_surprise_biasadd_readvariableop_resource:A
.model_1_sadness_matmul_readvariableop_resource:	?=
/model_1_sadness_biasadd_readvariableop_resource:=
*model_1_joy_matmul_readvariableop_resource:	?9
+model_1_joy_biasadd_readvariableop_resource:>
+model_1_fear_matmul_readvariableop_resource:	?:
,model_1_fear_biasadd_readvariableop_resource:A
.model_1_disgust_matmul_readvariableop_resource:	?=
/model_1_disgust_biasadd_readvariableop_resource:?
,model_1_anger_matmul_readvariableop_resource:	?;
-model_1_anger_biasadd_readvariableop_resource:F
3model_1_anticipation_matmul_readvariableop_resource:	?B
4model_1_anticipation_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??$model_1/Anger/BiasAdd/ReadVariableOp?#model_1/Anger/MatMul/ReadVariableOp?+model_1/Anticipation/BiasAdd/ReadVariableOp?*model_1/Anticipation/MatMul/ReadVariableOp?&model_1/Disgust/BiasAdd/ReadVariableOp?%model_1/Disgust/MatMul/ReadVariableOp?#model_1/Fear/BiasAdd/ReadVariableOp?"model_1/Fear/MatMul/ReadVariableOp?"model_1/Joy/BiasAdd/ReadVariableOp?!model_1/Joy/MatMul/ReadVariableOp?&model_1/Sadness/BiasAdd/ReadVariableOp?%model_1/Sadness/MatMul/ReadVariableOp?'model_1/Surprise/BiasAdd/ReadVariableOp?&model_1/Surprise/MatMul/ReadVariableOp?$model_1/Trust/BiasAdd/ReadVariableOp?#model_1/Trust/MatMul/ReadVariableOp?%model_1/conv1d/BiasAdd/ReadVariableOp?1model_1/conv1d/Conv1D/ExpandDims_1/ReadVariableOp?$model_1/embedding_1/embedding_lookup?
$model_1/embedding_1/embedding_lookupResourceGather+model_1_embedding_1_embedding_lookup_190034
main_input*
Tindices0*>
_class4
20loc:@model_1/embedding_1/embedding_lookup/190034*+
_output_shapes
:?????????@*
dtype0?
-model_1/embedding_1/embedding_lookup/IdentityIdentity-model_1/embedding_1/embedding_lookup:output:0*
T0*>
_class4
20loc:@model_1/embedding_1/embedding_lookup/190034*+
_output_shapes
:?????????@?
/model_1/embedding_1/embedding_lookup/Identity_1Identity6model_1/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????@o
$model_1/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
 model_1/conv1d/Conv1D/ExpandDims
ExpandDims8model_1/embedding_1/embedding_lookup/Identity_1:output:0-model_1/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@?
1model_1/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp:model_1_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype0h
&model_1/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
"model_1/conv1d/Conv1D/ExpandDims_1
ExpandDims9model_1/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0/model_1/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@??
model_1/conv1d/Conv1DConv2D)model_1/conv1d/Conv1D/ExpandDims:output:0+model_1/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
model_1/conv1d/Conv1D/SqueezeSqueezemodel_1/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

??????????
%model_1/conv1d/BiasAdd/ReadVariableOpReadVariableOp.model_1_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_1/conv1d/BiasAddBiasAdd&model_1/conv1d/Conv1D/Squeeze:output:0-model_1/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????s
model_1/conv1d/ReluRelumodel_1/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????t
2model_1/global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
 model_1/global_max_pooling1d/MaxMax!model_1/conv1d/Relu:activations:0;model_1/global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:???????????
model_1/dropout_39/IdentityIdentity)model_1/global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:???????????
#model_1/Trust/MatMul/ReadVariableOpReadVariableOp,model_1_trust_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Trust/MatMulMatMul$model_1/dropout_39/Identity:output:0+model_1/Trust/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model_1/Trust/BiasAdd/ReadVariableOpReadVariableOp-model_1_trust_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Trust/BiasAddBiasAddmodel_1/Trust/MatMul:product:0,model_1/Trust/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model_1/Trust/SigmoidSigmoidmodel_1/Trust/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
&model_1/Surprise/MatMul/ReadVariableOpReadVariableOp/model_1_surprise_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Surprise/MatMulMatMul$model_1/dropout_39/Identity:output:0.model_1/Surprise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
'model_1/Surprise/BiasAdd/ReadVariableOpReadVariableOp0model_1_surprise_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Surprise/BiasAddBiasAdd!model_1/Surprise/MatMul:product:0/model_1/Surprise/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????x
model_1/Surprise/SigmoidSigmoid!model_1/Surprise/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%model_1/Sadness/MatMul/ReadVariableOpReadVariableOp.model_1_sadness_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Sadness/MatMulMatMul$model_1/dropout_39/Identity:output:0-model_1/Sadness/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/Sadness/BiasAdd/ReadVariableOpReadVariableOp/model_1_sadness_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Sadness/BiasAddBiasAdd model_1/Sadness/MatMul:product:0.model_1/Sadness/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_1/Sadness/SigmoidSigmoid model_1/Sadness/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
!model_1/Joy/MatMul/ReadVariableOpReadVariableOp*model_1_joy_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Joy/MatMulMatMul$model_1/dropout_39/Identity:output:0)model_1/Joy/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
"model_1/Joy/BiasAdd/ReadVariableOpReadVariableOp+model_1_joy_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Joy/BiasAddBiasAddmodel_1/Joy/MatMul:product:0*model_1/Joy/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
model_1/Joy/SigmoidSigmoidmodel_1/Joy/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"model_1/Fear/MatMul/ReadVariableOpReadVariableOp+model_1_fear_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Fear/MatMulMatMul$model_1/dropout_39/Identity:output:0*model_1/Fear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#model_1/Fear/BiasAdd/ReadVariableOpReadVariableOp,model_1_fear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Fear/BiasAddBiasAddmodel_1/Fear/MatMul:product:0+model_1/Fear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
model_1/Fear/SigmoidSigmoidmodel_1/Fear/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%model_1/Disgust/MatMul/ReadVariableOpReadVariableOp.model_1_disgust_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Disgust/MatMulMatMul$model_1/dropout_39/Identity:output:0-model_1/Disgust/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_1/Disgust/BiasAdd/ReadVariableOpReadVariableOp/model_1_disgust_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Disgust/BiasAddBiasAdd model_1/Disgust/MatMul:product:0.model_1/Disgust/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_1/Disgust/SigmoidSigmoid model_1/Disgust/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
#model_1/Anger/MatMul/ReadVariableOpReadVariableOp,model_1_anger_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Anger/MatMulMatMul$model_1/dropout_39/Identity:output:0+model_1/Anger/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
$model_1/Anger/BiasAdd/ReadVariableOpReadVariableOp-model_1_anger_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Anger/BiasAddBiasAddmodel_1/Anger/MatMul:product:0,model_1/Anger/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
model_1/Anger/SigmoidSigmoidmodel_1/Anger/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
*model_1/Anticipation/MatMul/ReadVariableOpReadVariableOp3model_1_anticipation_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
model_1/Anticipation/MatMulMatMul$model_1/dropout_39/Identity:output:02model_1/Anticipation/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
+model_1/Anticipation/BiasAdd/ReadVariableOpReadVariableOp4model_1_anticipation_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_1/Anticipation/BiasAddBiasAdd%model_1/Anticipation/MatMul:product:03model_1/Anticipation/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
model_1/Anticipation/SigmoidSigmoid%model_1/Anticipation/BiasAdd:output:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitymodel_1/Anger/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity model_1/Anticipation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????l

Identity_2Identitymodel_1/Disgust/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????i

Identity_3Identitymodel_1/Fear/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????h

Identity_4Identitymodel_1/Joy/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????l

Identity_5Identitymodel_1/Sadness/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????m

Identity_6Identitymodel_1/Surprise/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????j

Identity_7Identitymodel_1/Trust/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp%^model_1/Anger/BiasAdd/ReadVariableOp$^model_1/Anger/MatMul/ReadVariableOp,^model_1/Anticipation/BiasAdd/ReadVariableOp+^model_1/Anticipation/MatMul/ReadVariableOp'^model_1/Disgust/BiasAdd/ReadVariableOp&^model_1/Disgust/MatMul/ReadVariableOp$^model_1/Fear/BiasAdd/ReadVariableOp#^model_1/Fear/MatMul/ReadVariableOp#^model_1/Joy/BiasAdd/ReadVariableOp"^model_1/Joy/MatMul/ReadVariableOp'^model_1/Sadness/BiasAdd/ReadVariableOp&^model_1/Sadness/MatMul/ReadVariableOp(^model_1/Surprise/BiasAdd/ReadVariableOp'^model_1/Surprise/MatMul/ReadVariableOp%^model_1/Trust/BiasAdd/ReadVariableOp$^model_1/Trust/MatMul/ReadVariableOp&^model_1/conv1d/BiasAdd/ReadVariableOp2^model_1/conv1d/Conv1D/ExpandDims_1/ReadVariableOp%^model_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2L
$model_1/Anger/BiasAdd/ReadVariableOp$model_1/Anger/BiasAdd/ReadVariableOp2J
#model_1/Anger/MatMul/ReadVariableOp#model_1/Anger/MatMul/ReadVariableOp2Z
+model_1/Anticipation/BiasAdd/ReadVariableOp+model_1/Anticipation/BiasAdd/ReadVariableOp2X
*model_1/Anticipation/MatMul/ReadVariableOp*model_1/Anticipation/MatMul/ReadVariableOp2P
&model_1/Disgust/BiasAdd/ReadVariableOp&model_1/Disgust/BiasAdd/ReadVariableOp2N
%model_1/Disgust/MatMul/ReadVariableOp%model_1/Disgust/MatMul/ReadVariableOp2J
#model_1/Fear/BiasAdd/ReadVariableOp#model_1/Fear/BiasAdd/ReadVariableOp2H
"model_1/Fear/MatMul/ReadVariableOp"model_1/Fear/MatMul/ReadVariableOp2H
"model_1/Joy/BiasAdd/ReadVariableOp"model_1/Joy/BiasAdd/ReadVariableOp2F
!model_1/Joy/MatMul/ReadVariableOp!model_1/Joy/MatMul/ReadVariableOp2P
&model_1/Sadness/BiasAdd/ReadVariableOp&model_1/Sadness/BiasAdd/ReadVariableOp2N
%model_1/Sadness/MatMul/ReadVariableOp%model_1/Sadness/MatMul/ReadVariableOp2R
'model_1/Surprise/BiasAdd/ReadVariableOp'model_1/Surprise/BiasAdd/ReadVariableOp2P
&model_1/Surprise/MatMul/ReadVariableOp&model_1/Surprise/MatMul/ReadVariableOp2L
$model_1/Trust/BiasAdd/ReadVariableOp$model_1/Trust/BiasAdd/ReadVariableOp2J
#model_1/Trust/MatMul/ReadVariableOp#model_1/Trust/MatMul/ReadVariableOp2N
%model_1/conv1d/BiasAdd/ReadVariableOp%model_1/conv1d/BiasAdd/ReadVariableOp2f
1model_1/conv1d/Conv1D/ExpandDims_1/ReadVariableOp1model_1/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2L
$model_1/embedding_1/embedding_lookup$model_1/embedding_1/embedding_lookup:S O
'
_output_shapes
:?????????
$
_user_specified_name
main_input
?
d
F__inference_dropout_39_layer_call_and_return_conditional_losses_191304

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_Anticipation_layer_call_fn_191325

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Anticipation_layer_call_and_return_conditional_losses_190317o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?	
C__inference_model_1_layer_call_and_return_conditional_losses_190331

inputs&
embedding_1_190148:
??@$
conv1d_190168:@?
conv1d_190170:	?
trust_190199:	?
trust_190201:"
surprise_190216:	?
surprise_190218:!
sadness_190233:	?
sadness_190235:

joy_190250:	?

joy_190252:
fear_190267:	?
fear_190269:!
disgust_190284:	?
disgust_190286:
anger_190301:	?
anger_190303:&
anticipation_190318:	?!
anticipation_190320:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Anger/StatefulPartitionedCall?$Anticipation/StatefulPartitionedCall?Disgust/StatefulPartitionedCall?Fear/StatefulPartitionedCall?Joy/StatefulPartitionedCall?Sadness/StatefulPartitionedCall? Surprise/StatefulPartitionedCall?Trust/StatefulPartitionedCall?conv1d/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_190148*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_190147?
conv1d/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0conv1d_190168conv1d_190170*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_190167?
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190178?
dropout_39/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_190185?
Trust/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0trust_190199trust_190201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Trust_layer_call_and_return_conditional_losses_190198?
 Surprise/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0surprise_190216surprise_190218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Surprise_layer_call_and_return_conditional_losses_190215?
Sadness/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0sadness_190233sadness_190235*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Sadness_layer_call_and_return_conditional_losses_190232?
Joy/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0
joy_190250
joy_190252*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Joy_layer_call_and_return_conditional_losses_190249?
Fear/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0fear_190267fear_190269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Fear_layer_call_and_return_conditional_losses_190266?
Disgust/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0disgust_190284disgust_190286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Disgust_layer_call_and_return_conditional_losses_190283?
Anger/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0anger_190301anger_190303*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Anger_layer_call_and_return_conditional_losses_190300?
$Anticipation/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0anticipation_190318anticipation_190320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Anticipation_layer_call_and_return_conditional_losses_190317|
IdentityIdentity-Anticipation/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_1Identity&Anger/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_2Identity(Disgust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????v

Identity_3Identity%Fear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_4Identity$Joy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_5Identity(Sadness/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????z

Identity_6Identity)Surprise/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_7Identity&Trust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Anger/StatefulPartitionedCall%^Anticipation/StatefulPartitionedCall ^Disgust/StatefulPartitionedCall^Fear/StatefulPartitionedCall^Joy/StatefulPartitionedCall ^Sadness/StatefulPartitionedCall!^Surprise/StatefulPartitionedCall^Trust/StatefulPartitionedCall^conv1d/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2>
Anger/StatefulPartitionedCallAnger/StatefulPartitionedCall2L
$Anticipation/StatefulPartitionedCall$Anticipation/StatefulPartitionedCall2B
Disgust/StatefulPartitionedCallDisgust/StatefulPartitionedCall2<
Fear/StatefulPartitionedCallFear/StatefulPartitionedCall2:
Joy/StatefulPartitionedCallJoy/StatefulPartitionedCall2B
Sadness/StatefulPartitionedCallSadness/StatefulPartitionedCall2D
 Surprise/StatefulPartitionedCall Surprise/StatefulPartitionedCall2>
Trust/StatefulPartitionedCallTrust/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?C
?	
C__inference_model_1_layer_call_and_return_conditional_losses_190866

main_input&
embedding_1_190808:
??@$
conv1d_190811:@?
conv1d_190813:	?
trust_190818:	?
trust_190820:"
surprise_190823:	?
surprise_190825:!
sadness_190828:	?
sadness_190830:

joy_190833:	?

joy_190835:
fear_190838:	?
fear_190840:!
disgust_190843:	?
disgust_190845:
anger_190848:	?
anger_190850:&
anticipation_190853:	?!
anticipation_190855:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Anger/StatefulPartitionedCall?$Anticipation/StatefulPartitionedCall?Disgust/StatefulPartitionedCall?Fear/StatefulPartitionedCall?Joy/StatefulPartitionedCall?Sadness/StatefulPartitionedCall? Surprise/StatefulPartitionedCall?Trust/StatefulPartitionedCall?conv1d/StatefulPartitionedCall?"dropout_39/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall
main_inputembedding_1_190808*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_190147?
conv1d/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0conv1d_190811conv1d_190813*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_190167?
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190178?
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_190486?
Trust/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0trust_190818trust_190820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Trust_layer_call_and_return_conditional_losses_190198?
 Surprise/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0surprise_190823surprise_190825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Surprise_layer_call_and_return_conditional_losses_190215?
Sadness/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0sadness_190828sadness_190830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Sadness_layer_call_and_return_conditional_losses_190232?
Joy/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0
joy_190833
joy_190835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Joy_layer_call_and_return_conditional_losses_190249?
Fear/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0fear_190838fear_190840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Fear_layer_call_and_return_conditional_losses_190266?
Disgust/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0disgust_190843disgust_190845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Disgust_layer_call_and_return_conditional_losses_190283?
Anger/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0anger_190848anger_190850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Anger_layer_call_and_return_conditional_losses_190300?
$Anticipation/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0anticipation_190853anticipation_190855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Anticipation_layer_call_and_return_conditional_losses_190317|
IdentityIdentity-Anticipation/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_1Identity&Anger/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_2Identity(Disgust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????v

Identity_3Identity%Fear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_4Identity$Joy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_5Identity(Sadness/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????z

Identity_6Identity)Surprise/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_7Identity&Trust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Anger/StatefulPartitionedCall%^Anticipation/StatefulPartitionedCall ^Disgust/StatefulPartitionedCall^Fear/StatefulPartitionedCall^Joy/StatefulPartitionedCall ^Sadness/StatefulPartitionedCall!^Surprise/StatefulPartitionedCall^Trust/StatefulPartitionedCall^conv1d/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2>
Anger/StatefulPartitionedCallAnger/StatefulPartitionedCall2L
$Anticipation/StatefulPartitionedCall$Anticipation/StatefulPartitionedCall2B
Disgust/StatefulPartitionedCallDisgust/StatefulPartitionedCall2<
Fear/StatefulPartitionedCallFear/StatefulPartitionedCall2:
Joy/StatefulPartitionedCallJoy/StatefulPartitionedCall2B
Sadness/StatefulPartitionedCallSadness/StatefulPartitionedCall2D
 Surprise/StatefulPartitionedCall Surprise/StatefulPartitionedCall2>
Trust/StatefulPartitionedCallTrust/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
main_input
?
?
(__inference_Disgust_layer_call_fn_191365

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Disgust_layer_call_and_return_conditional_losses_190283o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_Disgust_layer_call_and_return_conditional_losses_191376

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_Trust_layer_call_fn_191465

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Trust_layer_call_and_return_conditional_losses_190198o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
?__inference_Joy_layer_call_and_return_conditional_losses_190249

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_Sadness_layer_call_fn_191425

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Sadness_layer_call_and_return_conditional_losses_190232o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_190931

main_input
unknown:
??@ 
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
	unknown_4:	?
	unknown_5:
	unknown_6:	?
	unknown_7:
	unknown_8:	?
	unknown_9:

unknown_10:	?

unknown_11:

unknown_12:	?

unknown_13:

unknown_14:	?

unknown_15:

unknown_16:	?

unknown_17:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
main_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_190118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:?????????q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:?????????q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:?????????q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
main_input
?
?
B__inference_conv1d_layer_call_and_return_conditional_losses_191267

inputsB
+conv1d_expanddims_1_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@??
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
H__inference_Anticipation_layer_call_and_return_conditional_losses_190317

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_embedding_1_layer_call_fn_191233

inputs
unknown:
??@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_190147s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_39_layer_call_and_return_conditional_losses_190185

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_Surprise_layer_call_and_return_conditional_losses_190215

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_190988

inputs
unknown:
??@ 
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
	unknown_4:	?
	unknown_5:
	unknown_6:	?
	unknown_7:
	unknown_8:	?
	unknown_9:

unknown_10:	?

unknown_11:

unknown_12:	?

unknown_13:

unknown_14:	?

unknown_15:

unknown_16:	?

unknown_17:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_190331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:?????????q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:?????????q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:?????????q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Q
5__inference_global_max_pooling1d_layer_call_fn_191272

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190128i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
A__inference_Trust_layer_call_and_return_conditional_losses_190198

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_Fear_layer_call_fn_191385

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Fear_layer_call_and_return_conditional_losses_190266o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?6
"__inference__traced_restore_192092
file_prefix;
'assignvariableop_embedding_1_embeddings:
??@7
 assignvariableop_1_conv1d_kernel:@?-
assignvariableop_2_conv1d_bias:	?9
&assignvariableop_3_anticipation_kernel:	?2
$assignvariableop_4_anticipation_bias:2
assignvariableop_5_anger_kernel:	?+
assignvariableop_6_anger_bias:4
!assignvariableop_7_disgust_kernel:	?-
assignvariableop_8_disgust_bias:1
assignvariableop_9_fear_kernel:	?+
assignvariableop_10_fear_bias:1
assignvariableop_11_joy_kernel:	?*
assignvariableop_12_joy_bias:5
"assignvariableop_13_sadness_kernel:	?.
 assignvariableop_14_sadness_bias:6
#assignvariableop_15_surprise_kernel:	?/
!assignvariableop_16_surprise_bias:3
 assignvariableop_17_trust_kernel:	?,
assignvariableop_18_trust_bias:$
assignvariableop_19_beta_1: $
assignvariableop_20_beta_2: #
assignvariableop_21_decay: +
!assignvariableop_22_learning_rate: )
assignvariableop_23_adamax_iter:	 #
assignvariableop_24_total: #
assignvariableop_25_count: %
assignvariableop_26_total_1: %
assignvariableop_27_count_1: %
assignvariableop_28_total_2: %
assignvariableop_29_count_2: %
assignvariableop_30_total_3: %
assignvariableop_31_count_3: %
assignvariableop_32_total_4: %
assignvariableop_33_count_4: %
assignvariableop_34_total_5: %
assignvariableop_35_count_5: %
assignvariableop_36_total_6: %
assignvariableop_37_count_6: %
assignvariableop_38_total_7: %
assignvariableop_39_count_7: %
assignvariableop_40_total_8: %
assignvariableop_41_count_8: %
assignvariableop_42_total_9: %
assignvariableop_43_count_9: &
assignvariableop_44_total_10: &
assignvariableop_45_count_10: &
assignvariableop_46_total_11: &
assignvariableop_47_count_11: &
assignvariableop_48_total_12: &
assignvariableop_49_count_12: &
assignvariableop_50_total_13: &
assignvariableop_51_count_13: &
assignvariableop_52_total_14: &
assignvariableop_53_count_14: &
assignvariableop_54_total_15: &
assignvariableop_55_count_15: &
assignvariableop_56_total_16: &
assignvariableop_57_count_16: G
3assignvariableop_58_adamax_embedding_1_embeddings_m:
??@A
*assignvariableop_59_adamax_conv1d_kernel_m:@?7
(assignvariableop_60_adamax_conv1d_bias_m:	?C
0assignvariableop_61_adamax_anticipation_kernel_m:	?<
.assignvariableop_62_adamax_anticipation_bias_m:<
)assignvariableop_63_adamax_anger_kernel_m:	?5
'assignvariableop_64_adamax_anger_bias_m:>
+assignvariableop_65_adamax_disgust_kernel_m:	?7
)assignvariableop_66_adamax_disgust_bias_m:;
(assignvariableop_67_adamax_fear_kernel_m:	?4
&assignvariableop_68_adamax_fear_bias_m::
'assignvariableop_69_adamax_joy_kernel_m:	?3
%assignvariableop_70_adamax_joy_bias_m:>
+assignvariableop_71_adamax_sadness_kernel_m:	?7
)assignvariableop_72_adamax_sadness_bias_m:?
,assignvariableop_73_adamax_surprise_kernel_m:	?8
*assignvariableop_74_adamax_surprise_bias_m:<
)assignvariableop_75_adamax_trust_kernel_m:	?5
'assignvariableop_76_adamax_trust_bias_m:G
3assignvariableop_77_adamax_embedding_1_embeddings_v:
??@A
*assignvariableop_78_adamax_conv1d_kernel_v:@?7
(assignvariableop_79_adamax_conv1d_bias_v:	?C
0assignvariableop_80_adamax_anticipation_kernel_v:	?<
.assignvariableop_81_adamax_anticipation_bias_v:<
)assignvariableop_82_adamax_anger_kernel_v:	?5
'assignvariableop_83_adamax_anger_bias_v:>
+assignvariableop_84_adamax_disgust_kernel_v:	?7
)assignvariableop_85_adamax_disgust_bias_v:;
(assignvariableop_86_adamax_fear_kernel_v:	?4
&assignvariableop_87_adamax_fear_bias_v::
'assignvariableop_88_adamax_joy_kernel_v:	?3
%assignvariableop_89_adamax_joy_bias_v:>
+assignvariableop_90_adamax_sadness_kernel_v:	?7
)assignvariableop_91_adamax_sadness_bias_v:?
,assignvariableop_92_adamax_surprise_kernel_v:	?8
*assignvariableop_93_adamax_surprise_bias_v:<
)assignvariableop_94_adamax_trust_kernel_v:	?5
'assignvariableop_95_adamax_trust_bias_v:
identity_97??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?2
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?1
value?1B?1aB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?
value?B?aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*o
dtypese
c2a	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv1d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_anticipation_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_anticipation_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_anger_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_anger_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_disgust_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_disgust_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_fear_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_fear_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_joy_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_joy_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_sadness_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp assignvariableop_14_sadness_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_surprise_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_surprise_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp assignvariableop_17_trust_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_trust_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_adamax_iterIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_3Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_3Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_4Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_4Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_total_5Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOpassignvariableop_35_count_5Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOpassignvariableop_36_total_6Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOpassignvariableop_37_count_6Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOpassignvariableop_38_total_7Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOpassignvariableop_39_count_7Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOpassignvariableop_40_total_8Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOpassignvariableop_41_count_8Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOpassignvariableop_42_total_9Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOpassignvariableop_43_count_9Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOpassignvariableop_44_total_10Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOpassignvariableop_45_count_10Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOpassignvariableop_46_total_11Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_count_11Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpassignvariableop_48_total_12Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOpassignvariableop_49_count_12Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_13Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_13Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOpassignvariableop_52_total_14Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOpassignvariableop_53_count_14Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOpassignvariableop_54_total_15Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOpassignvariableop_55_count_15Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_16Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_16Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp3assignvariableop_58_adamax_embedding_1_embeddings_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adamax_conv1d_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adamax_conv1d_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp0assignvariableop_61_adamax_anticipation_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp.assignvariableop_62_adamax_anticipation_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adamax_anger_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adamax_anger_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adamax_disgust_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adamax_disgust_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp(assignvariableop_67_adamax_fear_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp&assignvariableop_68_adamax_fear_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp'assignvariableop_69_adamax_joy_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp%assignvariableop_70_adamax_joy_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adamax_sadness_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adamax_sadness_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adamax_surprise_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adamax_surprise_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adamax_trust_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adamax_trust_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp3assignvariableop_77_adamax_embedding_1_embeddings_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adamax_conv1d_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adamax_conv1d_bias_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp0assignvariableop_80_adamax_anticipation_kernel_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp.assignvariableop_81_adamax_anticipation_bias_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adamax_anger_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adamax_anger_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp+assignvariableop_84_adamax_disgust_kernel_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp)assignvariableop_85_adamax_disgust_bias_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adamax_fear_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp&assignvariableop_87_adamax_fear_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp'assignvariableop_88_adamax_joy_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp%assignvariableop_89_adamax_joy_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp+assignvariableop_90_adamax_sadness_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp)assignvariableop_91_adamax_sadness_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp,assignvariableop_92_adamax_surprise_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adamax_surprise_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adamax_trust_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp'assignvariableop_95_adamax_trust_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_96Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_97IdentityIdentity_96:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95*"
_acd_function_control_output(*
_output_shapes
 "#
identity_97Identity_97:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
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
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_95:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_191283

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_190744

main_input
unknown:
??@ 
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
	unknown_4:	?
	unknown_5:
	unknown_6:	?
	unknown_7:
	unknown_8:	?
	unknown_9:

unknown_10:	?

unknown_11:

unknown_12:	?

unknown_13:

unknown_14:	?

unknown_15:

unknown_16:	?

unknown_17:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
main_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_190632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:?????????q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:?????????q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:?????????q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
main_input
?
?
$__inference_Joy_layer_call_fn_191405

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Joy_layer_call_and_return_conditional_losses_190249o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_Anger_layer_call_and_return_conditional_losses_190300

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
+__inference_dropout_39_layer_call_fn_191299

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_190486p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?	
C__inference_model_1_layer_call_and_return_conditional_losses_190805

main_input&
embedding_1_190747:
??@$
conv1d_190750:@?
conv1d_190752:	?
trust_190757:	?
trust_190759:"
surprise_190762:	?
surprise_190764:!
sadness_190767:	?
sadness_190769:

joy_190772:	?

joy_190774:
fear_190777:	?
fear_190779:!
disgust_190782:	?
disgust_190784:
anger_190787:	?
anger_190789:&
anticipation_190792:	?!
anticipation_190794:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Anger/StatefulPartitionedCall?$Anticipation/StatefulPartitionedCall?Disgust/StatefulPartitionedCall?Fear/StatefulPartitionedCall?Joy/StatefulPartitionedCall?Sadness/StatefulPartitionedCall? Surprise/StatefulPartitionedCall?Trust/StatefulPartitionedCall?conv1d/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCall
main_inputembedding_1_190747*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_190147?
conv1d/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0conv1d_190750conv1d_190752*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_190167?
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190178?
dropout_39/PartitionedCallPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_190185?
Trust/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0trust_190757trust_190759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Trust_layer_call_and_return_conditional_losses_190198?
 Surprise/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0surprise_190762surprise_190764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Surprise_layer_call_and_return_conditional_losses_190215?
Sadness/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0sadness_190767sadness_190769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Sadness_layer_call_and_return_conditional_losses_190232?
Joy/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0
joy_190772
joy_190774*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Joy_layer_call_and_return_conditional_losses_190249?
Fear/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0fear_190777fear_190779*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Fear_layer_call_and_return_conditional_losses_190266?
Disgust/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0disgust_190782disgust_190784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Disgust_layer_call_and_return_conditional_losses_190283?
Anger/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0anger_190787anger_190789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Anger_layer_call_and_return_conditional_losses_190300?
$Anticipation/StatefulPartitionedCallStatefulPartitionedCall#dropout_39/PartitionedCall:output:0anticipation_190792anticipation_190794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Anticipation_layer_call_and_return_conditional_losses_190317|
IdentityIdentity-Anticipation/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_1Identity&Anger/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_2Identity(Disgust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????v

Identity_3Identity%Fear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_4Identity$Joy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_5Identity(Sadness/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????z

Identity_6Identity)Surprise/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_7Identity&Trust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Anger/StatefulPartitionedCall%^Anticipation/StatefulPartitionedCall ^Disgust/StatefulPartitionedCall^Fear/StatefulPartitionedCall^Joy/StatefulPartitionedCall ^Sadness/StatefulPartitionedCall!^Surprise/StatefulPartitionedCall^Trust/StatefulPartitionedCall^conv1d/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2>
Anger/StatefulPartitionedCallAnger/StatefulPartitionedCall2L
$Anticipation/StatefulPartitionedCall$Anticipation/StatefulPartitionedCall2B
Disgust/StatefulPartitionedCallDisgust/StatefulPartitionedCall2<
Fear/StatefulPartitionedCallFear/StatefulPartitionedCall2:
Joy/StatefulPartitionedCallJoy/StatefulPartitionedCall2B
Sadness/StatefulPartitionedCallSadness/StatefulPartitionedCall2D
 Surprise/StatefulPartitionedCall Surprise/StatefulPartitionedCall2>
Trust/StatefulPartitionedCallTrust/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
main_input
?

?
D__inference_Surprise_layer_call_and_return_conditional_losses_191456

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190128

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :m
MaxMaxinputsMax/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????]
IdentityIdentityMax:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
A__inference_Trust_layer_call_and_return_conditional_losses_191476

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?d
?
C__inference_model_1_layer_call_and_return_conditional_losses_191132

inputs7
#embedding_1_embedding_lookup_191048:
??@I
2conv1d_conv1d_expanddims_1_readvariableop_resource:@?5
&conv1d_biasadd_readvariableop_resource:	?7
$trust_matmul_readvariableop_resource:	?3
%trust_biasadd_readvariableop_resource::
'surprise_matmul_readvariableop_resource:	?6
(surprise_biasadd_readvariableop_resource:9
&sadness_matmul_readvariableop_resource:	?5
'sadness_biasadd_readvariableop_resource:5
"joy_matmul_readvariableop_resource:	?1
#joy_biasadd_readvariableop_resource:6
#fear_matmul_readvariableop_resource:	?2
$fear_biasadd_readvariableop_resource:9
&disgust_matmul_readvariableop_resource:	?5
'disgust_biasadd_readvariableop_resource:7
$anger_matmul_readvariableop_resource:	?3
%anger_biasadd_readvariableop_resource:>
+anticipation_matmul_readvariableop_resource:	?:
,anticipation_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Anger/BiasAdd/ReadVariableOp?Anger/MatMul/ReadVariableOp?#Anticipation/BiasAdd/ReadVariableOp?"Anticipation/MatMul/ReadVariableOp?Disgust/BiasAdd/ReadVariableOp?Disgust/MatMul/ReadVariableOp?Fear/BiasAdd/ReadVariableOp?Fear/MatMul/ReadVariableOp?Joy/BiasAdd/ReadVariableOp?Joy/MatMul/ReadVariableOp?Sadness/BiasAdd/ReadVariableOp?Sadness/MatMul/ReadVariableOp?Surprise/BiasAdd/ReadVariableOp?Surprise/MatMul/ReadVariableOp?Trust/BiasAdd/ReadVariableOp?Trust/MatMul/ReadVariableOp?conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?embedding_1/embedding_lookup?
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_191048inputs*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/191048*+
_output_shapes
:?????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/191048*+
_output_shapes
:?????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDims0embedding_1/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@?
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@??
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????u
dropout_39/IdentityIdentity!global_max_pooling1d/Max:output:0*
T0*(
_output_shapes
:???????????
Trust/MatMul/ReadVariableOpReadVariableOp$trust_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Trust/MatMulMatMuldropout_39/Identity:output:0#Trust/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
Trust/BiasAdd/ReadVariableOpReadVariableOp%trust_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Trust/BiasAddBiasAddTrust/MatMul:product:0$Trust/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
Trust/SigmoidSigmoidTrust/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Surprise/MatMul/ReadVariableOpReadVariableOp'surprise_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Surprise/MatMulMatMuldropout_39/Identity:output:0&Surprise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Surprise/BiasAdd/ReadVariableOpReadVariableOp(surprise_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Surprise/BiasAddBiasAddSurprise/MatMul:product:0'Surprise/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
Surprise/SigmoidSigmoidSurprise/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Sadness/MatMul/ReadVariableOpReadVariableOp&sadness_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Sadness/MatMulMatMuldropout_39/Identity:output:0%Sadness/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Sadness/BiasAdd/ReadVariableOpReadVariableOp'sadness_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Sadness/BiasAddBiasAddSadness/MatMul:product:0&Sadness/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
Sadness/SigmoidSigmoidSadness/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
Joy/MatMul/ReadVariableOpReadVariableOp"joy_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?

Joy/MatMulMatMuldropout_39/Identity:output:0!Joy/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
Joy/BiasAdd/ReadVariableOpReadVariableOp#joy_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Joy/BiasAddBiasAddJoy/MatMul:product:0"Joy/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Joy/SigmoidSigmoidJoy/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
Fear/MatMul/ReadVariableOpReadVariableOp#fear_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Fear/MatMulMatMuldropout_39/Identity:output:0"Fear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
Fear/BiasAdd/ReadVariableOpReadVariableOp$fear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Fear/BiasAddBiasAddFear/MatMul:product:0#Fear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
Fear/SigmoidSigmoidFear/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Disgust/MatMul/ReadVariableOpReadVariableOp&disgust_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Disgust/MatMulMatMuldropout_39/Identity:output:0%Disgust/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Disgust/BiasAdd/ReadVariableOpReadVariableOp'disgust_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Disgust/BiasAddBiasAddDisgust/MatMul:product:0&Disgust/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
Disgust/SigmoidSigmoidDisgust/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Anger/MatMul/ReadVariableOpReadVariableOp$anger_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Anger/MatMulMatMuldropout_39/Identity:output:0#Anger/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
Anger/BiasAdd/ReadVariableOpReadVariableOp%anger_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Anger/BiasAddBiasAddAnger/MatMul:product:0$Anger/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
Anger/SigmoidSigmoidAnger/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"Anticipation/MatMul/ReadVariableOpReadVariableOp+anticipation_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Anticipation/MatMulMatMuldropout_39/Identity:output:0*Anticipation/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Anticipation/BiasAdd/ReadVariableOpReadVariableOp,anticipation_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Anticipation/BiasAddBiasAddAnticipation/MatMul:product:0+Anticipation/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
Anticipation/SigmoidSigmoidAnticipation/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentityAnticipation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_1IdentityAnger/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????d

Identity_2IdentityDisgust/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????a

Identity_3IdentityFear/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????`

Identity_4IdentityJoy/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????d

Identity_5IdentitySadness/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????e

Identity_6IdentitySurprise/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_7IdentityTrust/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Anger/BiasAdd/ReadVariableOp^Anger/MatMul/ReadVariableOp$^Anticipation/BiasAdd/ReadVariableOp#^Anticipation/MatMul/ReadVariableOp^Disgust/BiasAdd/ReadVariableOp^Disgust/MatMul/ReadVariableOp^Fear/BiasAdd/ReadVariableOp^Fear/MatMul/ReadVariableOp^Joy/BiasAdd/ReadVariableOp^Joy/MatMul/ReadVariableOp^Sadness/BiasAdd/ReadVariableOp^Sadness/MatMul/ReadVariableOp ^Surprise/BiasAdd/ReadVariableOp^Surprise/MatMul/ReadVariableOp^Trust/BiasAdd/ReadVariableOp^Trust/MatMul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2<
Anger/BiasAdd/ReadVariableOpAnger/BiasAdd/ReadVariableOp2:
Anger/MatMul/ReadVariableOpAnger/MatMul/ReadVariableOp2J
#Anticipation/BiasAdd/ReadVariableOp#Anticipation/BiasAdd/ReadVariableOp2H
"Anticipation/MatMul/ReadVariableOp"Anticipation/MatMul/ReadVariableOp2@
Disgust/BiasAdd/ReadVariableOpDisgust/BiasAdd/ReadVariableOp2>
Disgust/MatMul/ReadVariableOpDisgust/MatMul/ReadVariableOp2:
Fear/BiasAdd/ReadVariableOpFear/BiasAdd/ReadVariableOp28
Fear/MatMul/ReadVariableOpFear/MatMul/ReadVariableOp28
Joy/BiasAdd/ReadVariableOpJoy/BiasAdd/ReadVariableOp26
Joy/MatMul/ReadVariableOpJoy/MatMul/ReadVariableOp2@
Sadness/BiasAdd/ReadVariableOpSadness/BiasAdd/ReadVariableOp2>
Sadness/MatMul/ReadVariableOpSadness/MatMul/ReadVariableOp2B
Surprise/BiasAdd/ReadVariableOpSurprise/BiasAdd/ReadVariableOp2@
Surprise/MatMul/ReadVariableOpSurprise/MatMul/ReadVariableOp2<
Trust/BiasAdd/ReadVariableOpTrust/BiasAdd/ReadVariableOp2:
Trust/MatMul/ReadVariableOpTrust/MatMul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
H__inference_Anticipation_layer_call_and_return_conditional_losses_191336

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
B__inference_conv1d_layer_call_and_return_conditional_losses_190167

inputsB
+conv1d_expanddims_1_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@?
"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@??
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

?????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:???????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
'__inference_conv1d_layer_call_fn_191251

inputs
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_190167t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_190386

main_input
unknown:
??@ 
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
	unknown_4:	?
	unknown_5:
	unknown_6:	?
	unknown_7:
	unknown_8:	?
	unknown_9:

unknown_10:	?

unknown_11:

unknown_12:	?

unknown_13:

unknown_14:	?

unknown_15:

unknown_16:	?

unknown_17:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
main_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_190331o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:?????????q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:?????????q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:?????????q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
main_input
?
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_191242

inputs+
embedding_lookup_191236:
??@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_191236inputs*
Tindices0**
_class 
loc:@embedding_lookup/191236*+
_output_shapes
:?????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/191236*+
_output_shapes
:?????????@?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????@w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_Sadness_layer_call_and_return_conditional_losses_191436

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_191045

inputs
unknown:
??@ 
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:
	unknown_4:	?
	unknown_5:
	unknown_6:	?
	unknown_7:
	unknown_8:	?
	unknown_9:

unknown_10:	?

unknown_11:

unknown_12:	?

unknown_13:

unknown_14:	?

unknown_15:

unknown_16:	?

unknown_17:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout

2*
_collective_manager_ids
 *?
_output_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????*5
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_190632o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:?????????q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:?????????q

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:?????????q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:?????????q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:?????????q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:?????????q

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
?__inference_Joy_layer_call_and_return_conditional_losses_191416

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?k
?
C__inference_model_1_layer_call_and_return_conditional_losses_191226

inputs7
#embedding_1_embedding_lookup_191135:
??@I
2conv1d_conv1d_expanddims_1_readvariableop_resource:@?5
&conv1d_biasadd_readvariableop_resource:	?7
$trust_matmul_readvariableop_resource:	?3
%trust_biasadd_readvariableop_resource::
'surprise_matmul_readvariableop_resource:	?6
(surprise_biasadd_readvariableop_resource:9
&sadness_matmul_readvariableop_resource:	?5
'sadness_biasadd_readvariableop_resource:5
"joy_matmul_readvariableop_resource:	?1
#joy_biasadd_readvariableop_resource:6
#fear_matmul_readvariableop_resource:	?2
$fear_biasadd_readvariableop_resource:9
&disgust_matmul_readvariableop_resource:	?5
'disgust_biasadd_readvariableop_resource:7
$anger_matmul_readvariableop_resource:	?3
%anger_biasadd_readvariableop_resource:>
+anticipation_matmul_readvariableop_resource:	?:
,anticipation_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Anger/BiasAdd/ReadVariableOp?Anger/MatMul/ReadVariableOp?#Anticipation/BiasAdd/ReadVariableOp?"Anticipation/MatMul/ReadVariableOp?Disgust/BiasAdd/ReadVariableOp?Disgust/MatMul/ReadVariableOp?Fear/BiasAdd/ReadVariableOp?Fear/MatMul/ReadVariableOp?Joy/BiasAdd/ReadVariableOp?Joy/MatMul/ReadVariableOp?Sadness/BiasAdd/ReadVariableOp?Sadness/MatMul/ReadVariableOp?Surprise/BiasAdd/ReadVariableOp?Surprise/MatMul/ReadVariableOp?Trust/BiasAdd/ReadVariableOp?Trust/MatMul/ReadVariableOp?conv1d/BiasAdd/ReadVariableOp?)conv1d/Conv1D/ExpandDims_1/ReadVariableOp?embedding_1/embedding_lookup?
embedding_1/embedding_lookupResourceGather#embedding_1_embedding_lookup_191135inputs*
Tindices0*6
_class,
*(loc:@embedding_1/embedding_lookup/191135*+
_output_shapes
:?????????@*
dtype0?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding_1/embedding_lookup/191135*+
_output_shapes
:?????????@?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
??????????
conv1d/Conv1D/ExpandDims
ExpandDims0embedding_1/embedding_lookup/Identity_1:output:0%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@?
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:@?*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:@??
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:??????????*
squeeze_dims

??????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:??????????l
*global_max_pooling1d/Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :?
global_max_pooling1d/MaxMaxconv1d/Relu:activations:03global_max_pooling1d/Max/reduction_indices:output:0*
T0*(
_output_shapes
:??????????]
dropout_39/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
dropout_39/dropout/MulMul!global_max_pooling1d/Max:output:0!dropout_39/dropout/Const:output:0*
T0*(
_output_shapes
:??????????i
dropout_39/dropout/ShapeShape!global_max_pooling1d/Max:output:0*
T0*
_output_shapes
:?
/dropout_39/dropout/random_uniform/RandomUniformRandomUniform!dropout_39/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0f
!dropout_39/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout_39/dropout/GreaterEqualGreaterEqual8dropout_39/dropout/random_uniform/RandomUniform:output:0*dropout_39/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:???????????
dropout_39/dropout/CastCast#dropout_39/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:???????????
dropout_39/dropout/Mul_1Muldropout_39/dropout/Mul:z:0dropout_39/dropout/Cast:y:0*
T0*(
_output_shapes
:???????????
Trust/MatMul/ReadVariableOpReadVariableOp$trust_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Trust/MatMulMatMuldropout_39/dropout/Mul_1:z:0#Trust/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
Trust/BiasAdd/ReadVariableOpReadVariableOp%trust_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Trust/BiasAddBiasAddTrust/MatMul:product:0$Trust/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
Trust/SigmoidSigmoidTrust/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Surprise/MatMul/ReadVariableOpReadVariableOp'surprise_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Surprise/MatMulMatMuldropout_39/dropout/Mul_1:z:0&Surprise/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Surprise/BiasAdd/ReadVariableOpReadVariableOp(surprise_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Surprise/BiasAddBiasAddSurprise/MatMul:product:0'Surprise/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
Surprise/SigmoidSigmoidSurprise/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Sadness/MatMul/ReadVariableOpReadVariableOp&sadness_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Sadness/MatMulMatMuldropout_39/dropout/Mul_1:z:0%Sadness/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Sadness/BiasAdd/ReadVariableOpReadVariableOp'sadness_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Sadness/BiasAddBiasAddSadness/MatMul:product:0&Sadness/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
Sadness/SigmoidSigmoidSadness/BiasAdd:output:0*
T0*'
_output_shapes
:?????????}
Joy/MatMul/ReadVariableOpReadVariableOp"joy_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?

Joy/MatMulMatMuldropout_39/dropout/Mul_1:z:0!Joy/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
Joy/BiasAdd/ReadVariableOpReadVariableOp#joy_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Joy/BiasAddBiasAddJoy/MatMul:product:0"Joy/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Joy/SigmoidSigmoidJoy/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
Fear/MatMul/ReadVariableOpReadVariableOp#fear_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Fear/MatMulMatMuldropout_39/dropout/Mul_1:z:0"Fear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????|
Fear/BiasAdd/ReadVariableOpReadVariableOp$fear_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Fear/BiasAddBiasAddFear/MatMul:product:0#Fear/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????`
Fear/SigmoidSigmoidFear/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Disgust/MatMul/ReadVariableOpReadVariableOp&disgust_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Disgust/MatMulMatMuldropout_39/dropout/Mul_1:z:0%Disgust/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Disgust/BiasAdd/ReadVariableOpReadVariableOp'disgust_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Disgust/BiasAddBiasAddDisgust/MatMul:product:0&Disgust/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
Disgust/SigmoidSigmoidDisgust/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Anger/MatMul/ReadVariableOpReadVariableOp$anger_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Anger/MatMulMatMuldropout_39/dropout/Mul_1:z:0#Anger/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
Anger/BiasAdd/ReadVariableOpReadVariableOp%anger_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Anger/BiasAddBiasAddAnger/MatMul:product:0$Anger/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????b
Anger/SigmoidSigmoidAnger/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"Anticipation/MatMul/ReadVariableOpReadVariableOp+anticipation_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
Anticipation/MatMulMatMuldropout_39/dropout/Mul_1:z:0*Anticipation/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Anticipation/BiasAdd/ReadVariableOpReadVariableOp,anticipation_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Anticipation/BiasAddBiasAddAnticipation/MatMul:product:0+Anticipation/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
Anticipation/SigmoidSigmoidAnticipation/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentityAnticipation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_1IdentityAnger/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????d

Identity_2IdentityDisgust/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????a

Identity_3IdentityFear/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????`

Identity_4IdentityJoy/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????d

Identity_5IdentitySadness/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????e

Identity_6IdentitySurprise/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????b

Identity_7IdentityTrust/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Anger/BiasAdd/ReadVariableOp^Anger/MatMul/ReadVariableOp$^Anticipation/BiasAdd/ReadVariableOp#^Anticipation/MatMul/ReadVariableOp^Disgust/BiasAdd/ReadVariableOp^Disgust/MatMul/ReadVariableOp^Fear/BiasAdd/ReadVariableOp^Fear/MatMul/ReadVariableOp^Joy/BiasAdd/ReadVariableOp^Joy/MatMul/ReadVariableOp^Sadness/BiasAdd/ReadVariableOp^Sadness/MatMul/ReadVariableOp ^Surprise/BiasAdd/ReadVariableOp^Surprise/MatMul/ReadVariableOp^Trust/BiasAdd/ReadVariableOp^Trust/MatMul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2<
Anger/BiasAdd/ReadVariableOpAnger/BiasAdd/ReadVariableOp2:
Anger/MatMul/ReadVariableOpAnger/MatMul/ReadVariableOp2J
#Anticipation/BiasAdd/ReadVariableOp#Anticipation/BiasAdd/ReadVariableOp2H
"Anticipation/MatMul/ReadVariableOp"Anticipation/MatMul/ReadVariableOp2@
Disgust/BiasAdd/ReadVariableOpDisgust/BiasAdd/ReadVariableOp2>
Disgust/MatMul/ReadVariableOpDisgust/MatMul/ReadVariableOp2:
Fear/BiasAdd/ReadVariableOpFear/BiasAdd/ReadVariableOp28
Fear/MatMul/ReadVariableOpFear/MatMul/ReadVariableOp28
Joy/BiasAdd/ReadVariableOpJoy/BiasAdd/ReadVariableOp26
Joy/MatMul/ReadVariableOpJoy/MatMul/ReadVariableOp2@
Sadness/BiasAdd/ReadVariableOpSadness/BiasAdd/ReadVariableOp2>
Sadness/MatMul/ReadVariableOpSadness/MatMul/ReadVariableOp2B
Surprise/BiasAdd/ReadVariableOpSurprise/BiasAdd/ReadVariableOp2@
Surprise/MatMul/ReadVariableOpSurprise/MatMul/ReadVariableOp2<
Trust/BiasAdd/ReadVariableOpTrust/BiasAdd/ReadVariableOp2:
Trust/MatMul/ReadVariableOpTrust/MatMul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
e
F__inference_dropout_39_layer_call_and_return_conditional_losses_191316

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
A__inference_Anger_layer_call_and_return_conditional_losses_191356

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_Surprise_layer_call_fn_191445

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Surprise_layer_call_and_return_conditional_losses_190215o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
@__inference_Fear_layer_call_and_return_conditional_losses_191396

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?C
?	
C__inference_model_1_layer_call_and_return_conditional_losses_190632

inputs&
embedding_1_190574:
??@$
conv1d_190577:@?
conv1d_190579:	?
trust_190584:	?
trust_190586:"
surprise_190589:	?
surprise_190591:!
sadness_190594:	?
sadness_190596:

joy_190599:	?

joy_190601:
fear_190604:	?
fear_190606:!
disgust_190609:	?
disgust_190611:
anger_190614:	?
anger_190616:&
anticipation_190619:	?!
anticipation_190621:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7??Anger/StatefulPartitionedCall?$Anticipation/StatefulPartitionedCall?Disgust/StatefulPartitionedCall?Fear/StatefulPartitionedCall?Joy/StatefulPartitionedCall?Sadness/StatefulPartitionedCall? Surprise/StatefulPartitionedCall?Trust/StatefulPartitionedCall?conv1d/StatefulPartitionedCall?"dropout_39/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_1_190574*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_embedding_1_layer_call_and_return_conditional_losses_190147?
conv1d/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0conv1d_190577conv1d_190579*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_conv1d_layer_call_and_return_conditional_losses_190167?
$global_max_pooling1d/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_190178?
"dropout_39/StatefulPartitionedCallStatefulPartitionedCall-global_max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_190486?
Trust/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0trust_190584trust_190586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Trust_layer_call_and_return_conditional_losses_190198?
 Surprise/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0surprise_190589surprise_190591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Surprise_layer_call_and_return_conditional_losses_190215?
Sadness/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0sadness_190594sadness_190596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Sadness_layer_call_and_return_conditional_losses_190232?
Joy/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0
joy_190599
joy_190601*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_Joy_layer_call_and_return_conditional_losses_190249?
Fear/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0fear_190604fear_190606*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_Fear_layer_call_and_return_conditional_losses_190266?
Disgust/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0disgust_190609disgust_190611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Disgust_layer_call_and_return_conditional_losses_190283?
Anger/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0anger_190614anger_190616*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Anger_layer_call_and_return_conditional_losses_190300?
$Anticipation/StatefulPartitionedCallStatefulPartitionedCall+dropout_39/StatefulPartitionedCall:output:0anticipation_190619anticipation_190621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Anticipation_layer_call_and_return_conditional_losses_190317|
IdentityIdentity-Anticipation/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_1Identity&Anger/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_2Identity(Disgust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????v

Identity_3Identity%Fear/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????u

Identity_4Identity$Joy/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????y

Identity_5Identity(Sadness/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????z

Identity_6Identity)Surprise/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????w

Identity_7Identity&Trust/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Anger/StatefulPartitionedCall%^Anticipation/StatefulPartitionedCall ^Disgust/StatefulPartitionedCall^Fear/StatefulPartitionedCall^Joy/StatefulPartitionedCall ^Sadness/StatefulPartitionedCall!^Surprise/StatefulPartitionedCall^Trust/StatefulPartitionedCall^conv1d/StatefulPartitionedCall#^dropout_39/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:?????????: : : : : : : : : : : : : : : : : : : 2>
Anger/StatefulPartitionedCallAnger/StatefulPartitionedCall2L
$Anticipation/StatefulPartitionedCall$Anticipation/StatefulPartitionedCall2B
Disgust/StatefulPartitionedCallDisgust/StatefulPartitionedCall2<
Fear/StatefulPartitionedCallFear/StatefulPartitionedCall2:
Joy/StatefulPartitionedCallJoy/StatefulPartitionedCall2B
Sadness/StatefulPartitionedCallSadness/StatefulPartitionedCall2D
 Surprise/StatefulPartitionedCall Surprise/StatefulPartitionedCall2>
Trust/StatefulPartitionedCallTrust/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2H
"dropout_39/StatefulPartitionedCall"dropout_39/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_Sadness_layer_call_and_return_conditional_losses_190232

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
̦
?$
__inference__traced_save_191794
file_prefix5
1savev2_embedding_1_embeddings_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop2
.savev2_anticipation_kernel_read_readvariableop0
,savev2_anticipation_bias_read_readvariableop+
'savev2_anger_kernel_read_readvariableop)
%savev2_anger_bias_read_readvariableop-
)savev2_disgust_kernel_read_readvariableop+
'savev2_disgust_bias_read_readvariableop*
&savev2_fear_kernel_read_readvariableop(
$savev2_fear_bias_read_readvariableop)
%savev2_joy_kernel_read_readvariableop'
#savev2_joy_bias_read_readvariableop-
)savev2_sadness_kernel_read_readvariableop+
'savev2_sadness_bias_read_readvariableop.
*savev2_surprise_kernel_read_readvariableop,
(savev2_surprise_bias_read_readvariableop+
'savev2_trust_kernel_read_readvariableop)
%savev2_trust_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop*
&savev2_adamax_iter_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_3_read_readvariableop&
"savev2_count_3_read_readvariableop&
"savev2_total_4_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_5_read_readvariableop&
"savev2_count_5_read_readvariableop&
"savev2_total_6_read_readvariableop&
"savev2_count_6_read_readvariableop&
"savev2_total_7_read_readvariableop&
"savev2_count_7_read_readvariableop&
"savev2_total_8_read_readvariableop&
"savev2_count_8_read_readvariableop&
"savev2_total_9_read_readvariableop&
"savev2_count_9_read_readvariableop'
#savev2_total_10_read_readvariableop'
#savev2_count_10_read_readvariableop'
#savev2_total_11_read_readvariableop'
#savev2_count_11_read_readvariableop'
#savev2_total_12_read_readvariableop'
#savev2_count_12_read_readvariableop'
#savev2_total_13_read_readvariableop'
#savev2_count_13_read_readvariableop'
#savev2_total_14_read_readvariableop'
#savev2_count_14_read_readvariableop'
#savev2_total_15_read_readvariableop'
#savev2_count_15_read_readvariableop'
#savev2_total_16_read_readvariableop'
#savev2_count_16_read_readvariableop>
:savev2_adamax_embedding_1_embeddings_m_read_readvariableop5
1savev2_adamax_conv1d_kernel_m_read_readvariableop3
/savev2_adamax_conv1d_bias_m_read_readvariableop;
7savev2_adamax_anticipation_kernel_m_read_readvariableop9
5savev2_adamax_anticipation_bias_m_read_readvariableop4
0savev2_adamax_anger_kernel_m_read_readvariableop2
.savev2_adamax_anger_bias_m_read_readvariableop6
2savev2_adamax_disgust_kernel_m_read_readvariableop4
0savev2_adamax_disgust_bias_m_read_readvariableop3
/savev2_adamax_fear_kernel_m_read_readvariableop1
-savev2_adamax_fear_bias_m_read_readvariableop2
.savev2_adamax_joy_kernel_m_read_readvariableop0
,savev2_adamax_joy_bias_m_read_readvariableop6
2savev2_adamax_sadness_kernel_m_read_readvariableop4
0savev2_adamax_sadness_bias_m_read_readvariableop7
3savev2_adamax_surprise_kernel_m_read_readvariableop5
1savev2_adamax_surprise_bias_m_read_readvariableop4
0savev2_adamax_trust_kernel_m_read_readvariableop2
.savev2_adamax_trust_bias_m_read_readvariableop>
:savev2_adamax_embedding_1_embeddings_v_read_readvariableop5
1savev2_adamax_conv1d_kernel_v_read_readvariableop3
/savev2_adamax_conv1d_bias_v_read_readvariableop;
7savev2_adamax_anticipation_kernel_v_read_readvariableop9
5savev2_adamax_anticipation_bias_v_read_readvariableop4
0savev2_adamax_anger_kernel_v_read_readvariableop2
.savev2_adamax_anger_bias_v_read_readvariableop6
2savev2_adamax_disgust_kernel_v_read_readvariableop4
0savev2_adamax_disgust_bias_v_read_readvariableop3
/savev2_adamax_fear_kernel_v_read_readvariableop1
-savev2_adamax_fear_bias_v_read_readvariableop2
.savev2_adamax_joy_kernel_v_read_readvariableop0
,savev2_adamax_joy_bias_v_read_readvariableop6
2savev2_adamax_sadness_kernel_v_read_readvariableop4
0savev2_adamax_sadness_bias_v_read_readvariableop7
3savev2_adamax_surprise_kernel_v_read_readvariableop5
1savev2_adamax_surprise_bias_v_read_readvariableop4
0savev2_adamax_trust_kernel_v_read_readvariableop2
.savev2_adamax_trust_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?2
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?1
value?1B?1aB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/10/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/11/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/12/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/13/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/14/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/15/count/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/total/.ATTRIBUTES/VARIABLE_VALUEB5keras_api/metrics/16/count/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*?
value?B?aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_embedding_1_embeddings_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop.savev2_anticipation_kernel_read_readvariableop,savev2_anticipation_bias_read_readvariableop'savev2_anger_kernel_read_readvariableop%savev2_anger_bias_read_readvariableop)savev2_disgust_kernel_read_readvariableop'savev2_disgust_bias_read_readvariableop&savev2_fear_kernel_read_readvariableop$savev2_fear_bias_read_readvariableop%savev2_joy_kernel_read_readvariableop#savev2_joy_bias_read_readvariableop)savev2_sadness_kernel_read_readvariableop'savev2_sadness_bias_read_readvariableop*savev2_surprise_kernel_read_readvariableop(savev2_surprise_bias_read_readvariableop'savev2_trust_kernel_read_readvariableop%savev2_trust_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop&savev2_adamax_iter_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_3_read_readvariableop"savev2_count_3_read_readvariableop"savev2_total_4_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_5_read_readvariableop"savev2_count_5_read_readvariableop"savev2_total_6_read_readvariableop"savev2_count_6_read_readvariableop"savev2_total_7_read_readvariableop"savev2_count_7_read_readvariableop"savev2_total_8_read_readvariableop"savev2_count_8_read_readvariableop"savev2_total_9_read_readvariableop"savev2_count_9_read_readvariableop#savev2_total_10_read_readvariableop#savev2_count_10_read_readvariableop#savev2_total_11_read_readvariableop#savev2_count_11_read_readvariableop#savev2_total_12_read_readvariableop#savev2_count_12_read_readvariableop#savev2_total_13_read_readvariableop#savev2_count_13_read_readvariableop#savev2_total_14_read_readvariableop#savev2_count_14_read_readvariableop#savev2_total_15_read_readvariableop#savev2_count_15_read_readvariableop#savev2_total_16_read_readvariableop#savev2_count_16_read_readvariableop:savev2_adamax_embedding_1_embeddings_m_read_readvariableop1savev2_adamax_conv1d_kernel_m_read_readvariableop/savev2_adamax_conv1d_bias_m_read_readvariableop7savev2_adamax_anticipation_kernel_m_read_readvariableop5savev2_adamax_anticipation_bias_m_read_readvariableop0savev2_adamax_anger_kernel_m_read_readvariableop.savev2_adamax_anger_bias_m_read_readvariableop2savev2_adamax_disgust_kernel_m_read_readvariableop0savev2_adamax_disgust_bias_m_read_readvariableop/savev2_adamax_fear_kernel_m_read_readvariableop-savev2_adamax_fear_bias_m_read_readvariableop.savev2_adamax_joy_kernel_m_read_readvariableop,savev2_adamax_joy_bias_m_read_readvariableop2savev2_adamax_sadness_kernel_m_read_readvariableop0savev2_adamax_sadness_bias_m_read_readvariableop3savev2_adamax_surprise_kernel_m_read_readvariableop1savev2_adamax_surprise_bias_m_read_readvariableop0savev2_adamax_trust_kernel_m_read_readvariableop.savev2_adamax_trust_bias_m_read_readvariableop:savev2_adamax_embedding_1_embeddings_v_read_readvariableop1savev2_adamax_conv1d_kernel_v_read_readvariableop/savev2_adamax_conv1d_bias_v_read_readvariableop7savev2_adamax_anticipation_kernel_v_read_readvariableop5savev2_adamax_anticipation_bias_v_read_readvariableop0savev2_adamax_anger_kernel_v_read_readvariableop.savev2_adamax_anger_bias_v_read_readvariableop2savev2_adamax_disgust_kernel_v_read_readvariableop0savev2_adamax_disgust_bias_v_read_readvariableop/savev2_adamax_fear_kernel_v_read_readvariableop-savev2_adamax_fear_bias_v_read_readvariableop.savev2_adamax_joy_kernel_v_read_readvariableop,savev2_adamax_joy_bias_v_read_readvariableop2savev2_adamax_sadness_kernel_v_read_readvariableop0savev2_adamax_sadness_bias_v_read_readvariableop3savev2_adamax_surprise_kernel_v_read_readvariableop1savev2_adamax_surprise_bias_v_read_readvariableop0savev2_adamax_trust_kernel_v_read_readvariableop.savev2_adamax_trust_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *o
dtypese
c2a	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :
??@:@?:?:	?::	?::	?::	?::	?::	?::	?::	?:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
??@:@?:?:	?::	?::	?::	?::	?::	?::	?::	?::
??@:@?:?:	?::	?::	?::	?::	?::	?::	?::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??@:)%
#
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 	

_output_shapes
::%
!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: :*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :&;"
 
_output_shapes
:
??@:)<%
#
_output_shapes
:@?:!=

_output_shapes	
:?:%>!

_output_shapes
:	?: ?

_output_shapes
::%@!

_output_shapes
:	?: A

_output_shapes
::%B!

_output_shapes
:	?: C

_output_shapes
::%D!

_output_shapes
:	?: E

_output_shapes
::%F!

_output_shapes
:	?: G

_output_shapes
::%H!

_output_shapes
:	?: I

_output_shapes
::%J!

_output_shapes
:	?: K

_output_shapes
::%L!

_output_shapes
:	?: M

_output_shapes
::&N"
 
_output_shapes
:
??@:)O%
#
_output_shapes
:@?:!P

_output_shapes	
:?:%Q!

_output_shapes
:	?: R

_output_shapes
::%S!

_output_shapes
:	?: T

_output_shapes
::%U!

_output_shapes
:	?: V

_output_shapes
::%W!

_output_shapes
:	?: X

_output_shapes
::%Y!

_output_shapes
:	?: Z

_output_shapes
::%[!

_output_shapes
:	?: \

_output_shapes
::%]!

_output_shapes
:	?: ^

_output_shapes
::%_!

_output_shapes
:	?: `

_output_shapes
::a

_output_shapes
: 
?
?
G__inference_embedding_1_layer_call_and_return_conditional_losses_190147

inputs+
embedding_lookup_190141:
??@
identity??embedding_lookup?
embedding_lookupResourceGatherembedding_lookup_190141inputs*
Tindices0**
_class 
loc:@embedding_lookup/190141*+
_output_shapes
:?????????@*
dtype0?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0**
_class 
loc:@embedding_lookup/190141*+
_output_shapes
:?????????@?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????@w
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????@Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_39_layer_call_fn_191294

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_39_layer_call_and_return_conditional_losses_190185a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
l
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_191289

inputs
identityW
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
MaxMaxinputsMax/reduction_indices:output:0*
T0*(
_output_shapes
:??????????U
IdentityIdentityMax:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A

main_input3
serving_default_main_input:0?????????9
Anger0
StatefulPartitionedCall:0?????????@
Anticipation0
StatefulPartitionedCall:1?????????;
Disgust0
StatefulPartitionedCall:2?????????8
Fear0
StatefulPartitionedCall:3?????????7
Joy0
StatefulPartitionedCall:4?????????;
Sadness0
StatefulPartitionedCall:5?????????<
Surprise0
StatefulPartitionedCall:6?????????9
Trust0
StatefulPartitionedCall:7?????????tensorflow/serving/predict:̆
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
	optimizer
loss
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$	variables
%trainable_variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Xbeta_1

Ybeta_2
	Zdecay
[learning_rate
\iterm?m?m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?v?v?v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?"
	optimizer
 "
trackable_dict_wrapper
?
0
1
2
(3
)4
.5
/6
47
58
:9
;10
@11
A12
F13
G14
L15
M16
R17
S18"
trackable_list_wrapper
?
0
1
2
(3
)4
.5
/6
47
58
:9
;10
@11
A12
F13
G14
L15
M16
R17
S18"
trackable_list_wrapper
 "
trackable_list_wrapper
?
]non_trainable_variables

^layers
_metrics
`layer_regularization_losses
alayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(
??@2embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables

clayers
dmetrics
elayer_regularization_losses
flayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"@?2conv1d/kernel
:?2conv1d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
lnon_trainable_variables

mlayers
nmetrics
olayer_regularization_losses
player_metrics
 	variables
!trainable_variables
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
$	variables
%trainable_variables
&regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$	?2Anticipation/kernel
:2Anticipation/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
*	variables
+trainable_variables
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2Anger/kernel
:2
Anger/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2Disgust/kernel
:2Disgust/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
6	variables
7trainable_variables
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2Fear/kernel
:2	Fear/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
<	variables
=trainable_variables
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2
Joy/kernel
:2Joy/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2Sadness/kernel
:2Sadness/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?2Surprise/kernel
:2Surprise/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?2Trust/kernel
:2
Trust/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2Adamax/iter
 "
trackable_list_wrapper
~
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
10
11
12"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
1:/
??@2Adamax/embedding_1/embeddings/m
+:)@?2Adamax/conv1d/kernel/m
!:?2Adamax/conv1d/bias/m
-:+	?2Adamax/Anticipation/kernel/m
&:$2Adamax/Anticipation/bias/m
&:$	?2Adamax/Anger/kernel/m
:2Adamax/Anger/bias/m
(:&	?2Adamax/Disgust/kernel/m
!:2Adamax/Disgust/bias/m
%:#	?2Adamax/Fear/kernel/m
:2Adamax/Fear/bias/m
$:"	?2Adamax/Joy/kernel/m
:2Adamax/Joy/bias/m
(:&	?2Adamax/Sadness/kernel/m
!:2Adamax/Sadness/bias/m
):'	?2Adamax/Surprise/kernel/m
": 2Adamax/Surprise/bias/m
&:$	?2Adamax/Trust/kernel/m
:2Adamax/Trust/bias/m
1:/
??@2Adamax/embedding_1/embeddings/v
+:)@?2Adamax/conv1d/kernel/v
!:?2Adamax/conv1d/bias/v
-:+	?2Adamax/Anticipation/kernel/v
&:$2Adamax/Anticipation/bias/v
&:$	?2Adamax/Anger/kernel/v
:2Adamax/Anger/bias/v
(:&	?2Adamax/Disgust/kernel/v
!:2Adamax/Disgust/bias/v
%:#	?2Adamax/Fear/kernel/v
:2Adamax/Fear/bias/v
$:"	?2Adamax/Joy/kernel/v
:2Adamax/Joy/bias/v
(:&	?2Adamax/Sadness/kernel/v
!:2Adamax/Sadness/bias/v
):'	?2Adamax/Surprise/kernel/v
": 2Adamax/Surprise/bias/v
&:$	?2Adamax/Trust/kernel/v
:2Adamax/Trust/bias/v
?2?
(__inference_model_1_layer_call_fn_190386
(__inference_model_1_layer_call_fn_190988
(__inference_model_1_layer_call_fn_191045
(__inference_model_1_layer_call_fn_190744?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_191132
C__inference_model_1_layer_call_and_return_conditional_losses_191226
C__inference_model_1_layer_call_and_return_conditional_losses_190805
C__inference_model_1_layer_call_and_return_conditional_losses_190866?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_190118
main_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_embedding_1_layer_call_fn_191233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_embedding_1_layer_call_and_return_conditional_losses_191242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv1d_layer_call_fn_191251?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1d_layer_call_and_return_conditional_losses_191267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_global_max_pooling1d_layer_call_fn_191272
5__inference_global_max_pooling1d_layer_call_fn_191277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_191283
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_191289?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dropout_39_layer_call_fn_191294
+__inference_dropout_39_layer_call_fn_191299?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_39_layer_call_and_return_conditional_losses_191304
F__inference_dropout_39_layer_call_and_return_conditional_losses_191316?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_Anticipation_layer_call_fn_191325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_Anticipation_layer_call_and_return_conditional_losses_191336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Anger_layer_call_fn_191345?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Anger_layer_call_and_return_conditional_losses_191356?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Disgust_layer_call_fn_191365?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Disgust_layer_call_and_return_conditional_losses_191376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_Fear_layer_call_fn_191385?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_Fear_layer_call_and_return_conditional_losses_191396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
$__inference_Joy_layer_call_fn_191405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_Joy_layer_call_and_return_conditional_losses_191416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Sadness_layer_call_fn_191425?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Sadness_layer_call_and_return_conditional_losses_191436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_Surprise_layer_call_fn_191445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_Surprise_layer_call_and_return_conditional_losses_191456?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Trust_layer_call_fn_191465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Trust_layer_call_and_return_conditional_losses_191476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_190931
main_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
A__inference_Anger_layer_call_and_return_conditional_losses_191356]./0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_Anger_layer_call_fn_191345P./0?-
&?#
!?
inputs??????????
? "???????????
H__inference_Anticipation_layer_call_and_return_conditional_losses_191336]()0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
-__inference_Anticipation_layer_call_fn_191325P()0?-
&?#
!?
inputs??????????
? "???????????
C__inference_Disgust_layer_call_and_return_conditional_losses_191376]450?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_Disgust_layer_call_fn_191365P450?-
&?#
!?
inputs??????????
? "???????????
@__inference_Fear_layer_call_and_return_conditional_losses_191396]:;0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? y
%__inference_Fear_layer_call_fn_191385P:;0?-
&?#
!?
inputs??????????
? "???????????
?__inference_Joy_layer_call_and_return_conditional_losses_191416]@A0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? x
$__inference_Joy_layer_call_fn_191405P@A0?-
&?#
!?
inputs??????????
? "???????????
C__inference_Sadness_layer_call_and_return_conditional_losses_191436]FG0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? |
(__inference_Sadness_layer_call_fn_191425PFG0?-
&?#
!?
inputs??????????
? "???????????
D__inference_Surprise_layer_call_and_return_conditional_losses_191456]LM0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? }
)__inference_Surprise_layer_call_fn_191445PLM0?-
&?#
!?
inputs??????????
? "???????????
A__inference_Trust_layer_call_and_return_conditional_losses_191476]RS0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? z
&__inference_Trust_layer_call_fn_191465PRS0?-
&?#
!?
inputs??????????
? "???????????
!__inference__wrapped_model_190118?RSLMFG@A:;45./()3?0
)?&
$?!

main_input?????????
? "???
(
Anger?
Anger?????????
6
Anticipation&?#
Anticipation?????????
,
Disgust!?
Disgust?????????
&
Fear?
Fear?????????
$
Joy?
Joy?????????
,
Sadness!?
Sadness?????????
.
Surprise"?
Surprise?????????
(
Trust?
Trust??????????
B__inference_conv1d_layer_call_and_return_conditional_losses_191267e3?0
)?&
$?!
inputs?????????@
? "*?'
 ?
0??????????
? ?
'__inference_conv1d_layer_call_fn_191251X3?0
)?&
$?!
inputs?????????@
? "????????????
F__inference_dropout_39_layer_call_and_return_conditional_losses_191304^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_39_layer_call_and_return_conditional_losses_191316^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_39_layer_call_fn_191294Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_39_layer_call_fn_191299Q4?1
*?'
!?
inputs??????????
p
? "????????????
G__inference_embedding_1_layer_call_and_return_conditional_losses_191242_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????@
? ?
,__inference_embedding_1_layer_call_fn_191233R/?,
%?"
 ?
inputs?????????
? "??????????@?
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_191283wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+
$?!
0??????????????????
? ?
P__inference_global_max_pooling1d_layer_call_and_return_conditional_losses_191289^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????
? ?
5__inference_global_max_pooling1d_layer_call_fn_191272jE?B
;?8
6?3
inputs'???????????????????????????
? "!????????????????????
5__inference_global_max_pooling1d_layer_call_fn_191277Q4?1
*?'
%?"
inputs??????????
? "????????????
C__inference_model_1_layer_call_and_return_conditional_losses_190805?RSLMFG@A:;45./();?8
1?.
$?!

main_input?????????
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_190866?RSLMFG@A:;45./();?8
1?.
$?!

main_input?????????
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_191132?RSLMFG@A:;45./()7?4
-?*
 ?
inputs?????????
p 

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_191226?RSLMFG@A:;45./()7?4
-?*
 ?
inputs?????????
p

 
? "???
???
?
0/0?????????
?
0/1?????????
?
0/2?????????
?
0/3?????????
?
0/4?????????
?
0/5?????????
?
0/6?????????
?
0/7?????????
? ?
(__inference_model_1_layer_call_fn_190386?RSLMFG@A:;45./();?8
1?.
$?!

main_input?????????
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7??????????
(__inference_model_1_layer_call_fn_190744?RSLMFG@A:;45./();?8
1?.
$?!

main_input?????????
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7??????????
(__inference_model_1_layer_call_fn_190988?RSLMFG@A:;45./()7?4
-?*
 ?
inputs?????????
p 

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7??????????
(__inference_model_1_layer_call_fn_191045?RSLMFG@A:;45./()7?4
-?*
 ?
inputs?????????
p

 
? "???
?
0?????????
?
1?????????
?
2?????????
?
3?????????
?
4?????????
?
5?????????
?
6?????????
?
7??????????
$__inference_signature_wrapper_190931?RSLMFG@A:;45./()A?>
? 
7?4
2

main_input$?!

main_input?????????"???
(
Anger?
Anger?????????
6
Anticipation&?#
Anticipation?????????
,
Disgust!?
Disgust?????????
&
Fear?
Fear?????????
$
Joy?
Joy?????????
,
Sadness!?
Sadness?????????
.
Surprise"?
Surprise?????????
(
Trust?
Trust?????????