
F
x_inputPlaceholder*!
shape:����������*
dtype0
J
Reshape/shapeConst*%
valueB"����   �      *
dtype0
A
ReshapeReshapex_inputReshape/shape*
T0*
Tshape0
E
PlaceholderPlaceholder*
shape:���������*
dtype0
S
truncated_normal/shapeConst*%
valueB"            *
dtype0
B
truncated_normal/meanConst*
valueB
 *    *
dtype0
D
truncated_normal/stddevConst*
valueB
 *���=*
dtype0
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
seed2 *
T0*

seed *
dtype0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
d
Variable
VariableV2*
shape:*
dtype0*
	container *
shared_name 
�
Variable/AssignAssignVariabletruncated_normal*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
6
ConstConst*
valueB<*    *
dtype0
Z

Variable_1
VariableV2*
shape:<*
dtype0*
	container *
shared_name 

Variable_1/AssignAssign
Variable_1Const*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
L
depthwise/ShapeConst*%
valueB"            *
dtype0
L
depthwise/dilation_rateConst*
valueB"      *
dtype0
�
	depthwiseDepthwiseConv2dNativeReshapeVariable/read*
T0*
paddingVALID*
strides
*
data_formatNHWC
/
AddAdd	depthwiseVariable_1/read*
T0

ReluReluAdd*
T0
t
MaxPoolMaxPoolRelu*
T0*
paddingVALID*
data_formatNHWC*
strides
*
ksize

U
truncated_normal_1/shapeConst*%
valueB"      <      *
dtype0
D
truncated_normal_1/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_1/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
seed2 *
T0*

seed *
dtype0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
f

Variable_2
VariableV2*
shape:<*
dtype0*
	container *
shared_name 
�
Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
O
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2
8
Const_1Const*
valueBx*    *
dtype0
Z

Variable_3
VariableV2*
shape:x*
dtype0*
	container *
shared_name 
�
Variable_3/AssignAssign
Variable_3Const_1*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
O
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3
N
depthwise_1/ShapeConst*%
valueB"      <      *
dtype0
N
depthwise_1/dilation_rateConst*
valueB"      *
dtype0
�
depthwise_1DepthwiseConv2dNativeMaxPoolVariable_2/read*
T0*
paddingVALID*
strides
*
data_formatNHWC
3
Add_1Adddepthwise_1Variable_3/read*
T0

Relu_1ReluAdd_1*
T0
D
Reshape_1/shapeConst*
valueB"����h  *
dtype0
D
	Reshape_1ReshapeRelu_1Reshape_1/shape*
T0*
Tshape0
M
truncated_normal_2/shapeConst*
valueB"h  d   *
dtype0
D
truncated_normal_2/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_2/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
seed2 *
T0*

seed *
dtype0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
_

Variable_4
VariableV2*
shape:	� d*
dtype0*
	container *
shared_name 
�
Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
O
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4
8
Const_2Const*
valueBd*    *
dtype0
Z

Variable_5
VariableV2*
shape:d*
dtype0*
	container *
shared_name 
�
Variable_5/AssignAssign
Variable_5Const_2*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
O
Variable_5/readIdentity
Variable_5*
T0*
_class
loc:@Variable_5
[
MatMulMatMul	Reshape_1Variable_4/read*
T0*
transpose_b( *
transpose_a( 
.
Add_2AddMatMulVariable_5/read*
T0

TanhTanhAdd_2*
T0
M
truncated_normal_3/shapeConst*
valueB"d      *
dtype0
D
truncated_normal_3/meanConst*
valueB
 *    *
dtype0
F
truncated_normal_3/stddevConst*
valueB
 *���=*
dtype0
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
seed2 *
T0*

seed *
dtype0
e
truncated_normal_3/mulMul"truncated_normal_3/TruncatedNormaltruncated_normal_3/stddev*
T0
S
truncated_normal_3Addtruncated_normal_3/multruncated_normal_3/mean*
T0
^

Variable_6
VariableV2*
shape
:d*
dtype0*
	container *
shared_name 
�
Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
O
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6
8
Const_3Const*
valueB*    *
dtype0
Z

Variable_7
VariableV2*
shape:*
dtype0*
	container *
shared_name 
�
Variable_7/AssignAssign
Variable_7Const_3*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
O
Variable_7/readIdentity
Variable_7*
T0*
_class
loc:@Variable_7
X
MatMul_1MatMulTanhVariable_6/read*
T0*
transpose_b( *
transpose_a( 
.
addAddMatMul_1Variable_7/read*
T0
&
labels_outputSoftmaxadd*
T0
"
LogLoglabels_output*
T0
%
mulMulPlaceholderLog*
T0
<
Const_4Const*
valueB"       *
dtype0
>
SumSummulConst_4*
T0*

Tidx0*
	keep_dims( 

NegNegSum*
T0
8
gradients/ShapeConst*
valueB *
dtype0
<
gradients/ConstConst*
valueB
 *  �?*
dtype0
A
gradients/FillFillgradients/Shapegradients/Const*
T0
6
gradients/Neg_grad/NegNeggradients/Fill*
T0
U
 gradients/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0
v
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
T0*
Tshape0
?
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0
p
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*

Tmultiples0
G
gradients/mul_grad/ShapeShapePlaceholder*
T0*
out_type0
A
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
D
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
N
gradients/mul_grad/mul_1MulPlaceholdergradients/Sum_grad/Tile*
T0
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
�
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
�
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
s
gradients/Log_grad/Reciprocal
Reciprocallabels_output.^gradients/mul_grad/tuple/control_dependency_1*
T0
t
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0
W
 gradients/labels_output_grad/mulMulgradients/Log_grad/mullabels_output*
T0
`
2gradients/labels_output_grad/Sum/reduction_indicesConst*
valueB:*
dtype0
�
 gradients/labels_output_grad/SumSum gradients/labels_output_grad/mul2gradients/labels_output_grad/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
_
*gradients/labels_output_grad/Reshape/shapeConst*
valueB"����   *
dtype0
�
$gradients/labels_output_grad/ReshapeReshape gradients/labels_output_grad/Sum*gradients/labels_output_grad/Reshape/shape*
T0*
Tshape0
n
 gradients/labels_output_grad/subSubgradients/Log_grad/mul$gradients/labels_output_grad/Reshape*
T0
c
"gradients/labels_output_grad/mul_1Mul gradients/labels_output_grad/sublabels_output*
T0
D
gradients/add_grad/ShapeShapeMatMul_1*
T0*
out_type0
H
gradients/add_grad/Shape_1Const*
valueB:*
dtype0
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0
�
gradients/add_grad/SumSum"gradients/labels_output_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0
�
gradients/add_grad/Sum_1Sum"gradients/labels_output_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
�
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
�
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1
�
gradients/MatMul_1_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
T0*
transpose_b(*
transpose_a( 
�
 gradients/MatMul_1_grad/MatMul_1MatMulTanh+gradients/add_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
�
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
�
2gradients/MatMul_1_grad/tuple/control_dependency_1Identity gradients/MatMul_1_grad/MatMul_1)^gradients/MatMul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/MatMul_1_grad/MatMul_1
i
gradients/Tanh_grad/TanhGradTanhGradTanh0gradients/MatMul_1_grad/tuple/control_dependency*
T0
D
gradients/Add_2_grad/ShapeShapeMatMul*
T0*
out_type0
J
gradients/Add_2_grad/Shape_1Const*
valueB:d*
dtype0
�
*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0
�
gradients/Add_2_grad/SumSumgradients/Tanh_grad/TanhGrad*gradients/Add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0
�
gradients/Add_2_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
�
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
�
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1
�
gradients/MatMul_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
T0*
transpose_b(*
transpose_a( 
�
gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/Add_2_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
�
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
�
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
H
gradients/Reshape_1_grad/ShapeShapeRelu_1*
T0*
out_type0
�
 gradients/Reshape_1_grad/ReshapeReshape.gradients/MatMul_grad/tuple/control_dependencygradients/Reshape_1_grad/Shape*
T0*
Tshape0
]
gradients/Relu_1_grad/ReluGradReluGrad gradients/Reshape_1_grad/ReshapeRelu_1*
T0
I
gradients/Add_1_grad/ShapeShapedepthwise_1*
T0*
out_type0
J
gradients/Add_1_grad/Shape_1Const*
valueB:x*
dtype0
�
*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0
�
gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
t
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0
�
gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
z
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
�
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
�
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1
K
 gradients/depthwise_1_grad/ShapeShapeMaxPool*
T0*
out_type0
�
=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput gradients/depthwise_1_grad/ShapeVariable_2/read-gradients/Add_1_grad/tuple/control_dependency*
T0*
paddingVALID*
strides
*
data_formatNHWC
_
"gradients/depthwise_1_grad/Shape_1Const*%
valueB"      <      *
dtype0
�
>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterMaxPool"gradients/depthwise_1_grad/Shape_1-gradients/Add_1_grad/tuple/control_dependency*
T0*
paddingVALID*
strides
*
data_formatNHWC
�
+gradients/depthwise_1_grad/tuple/group_depsNoOp>^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput?^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter
�
3gradients/depthwise_1_grad/tuple/control_dependencyIdentity=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput,^gradients/depthwise_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput
�
5gradients/depthwise_1_grad/tuple/control_dependency_1Identity>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter,^gradients/depthwise_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter
�
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/depthwise_1_grad/tuple/control_dependency*
strides
*
paddingVALID*
T0*
ksize
*
data_formatNHWC
[
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0
E
gradients/Add_grad/ShapeShape	depthwise*
T0*
out_type0
H
gradients/Add_grad/Shape_1Const*
valueB:<*
dtype0
�
(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0
�
gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
n
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0
�
gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
t
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
�
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape
�
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1
I
gradients/depthwise_grad/ShapeShapeReshape*
T0*
out_type0
�
;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInputgradients/depthwise_grad/ShapeVariable/read+gradients/Add_grad/tuple/control_dependency*
T0*
paddingVALID*
strides
*
data_formatNHWC
]
 gradients/depthwise_grad/Shape_1Const*%
valueB"            *
dtype0
�
<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterReshape gradients/depthwise_grad/Shape_1+gradients/Add_grad/tuple/control_dependency*
T0*
paddingVALID*
strides
*
data_formatNHWC
�
)gradients/depthwise_grad/tuple/group_depsNoOp<^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput=^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
�
1gradients/depthwise_grad/tuple/control_dependencyIdentity;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*^gradients/depthwise_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput
�
3gradients/depthwise_grad/tuple/control_dependency_1Identity<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*^gradients/depthwise_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
c
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@Variable*
dtype0
t
beta1_power
VariableV2*
shape: *
dtype0*
	container *
_class
loc:@Variable*
shared_name 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
O
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable
c
beta2_power/initial_valueConst*
valueB
 *w�?*
_class
loc:@Variable*
dtype0
t
beta2_power
VariableV2*
shape: *
dtype0*
	container *
_class
loc:@Variable*
shared_name 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
O
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable
y
Variable/Adam/Initializer/zerosConst*%
valueB*    *
_class
loc:@Variable*
dtype0
�
Variable/Adam
VariableV2*
shape:*
dtype0*
	container *
_class
loc:@Variable*
shared_name 
�
Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
S
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable
{
!Variable/Adam_1/Initializer/zerosConst*%
valueB*    *
_class
loc:@Variable*
dtype0
�
Variable/Adam_1
VariableV2*
shape:*
dtype0*
	container *
_class
loc:@Variable*
shared_name 
�
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
W
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable
q
!Variable_1/Adam/Initializer/zerosConst*
valueB<*    *
_class
loc:@Variable_1*
dtype0
~
Variable_1/Adam
VariableV2*
shape:<*
dtype0*
	container *
_class
loc:@Variable_1*
shared_name 
�
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
Y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1
s
#Variable_1/Adam_1/Initializer/zerosConst*
valueB<*    *
_class
loc:@Variable_1*
dtype0
�
Variable_1/Adam_1
VariableV2*
shape:<*
dtype0*
	container *
_class
loc:@Variable_1*
shared_name 
�
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
]
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1
}
!Variable_2/Adam/Initializer/zerosConst*%
valueB<*    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam
VariableV2*
shape:<*
dtype0*
	container *
_class
loc:@Variable_2*
shared_name 
�
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
Y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2

#Variable_2/Adam_1/Initializer/zerosConst*%
valueB<*    *
_class
loc:@Variable_2*
dtype0
�
Variable_2/Adam_1
VariableV2*
shape:<*
dtype0*
	container *
_class
loc:@Variable_2*
shared_name 
�
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
]
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2
q
!Variable_3/Adam/Initializer/zerosConst*
valueBx*    *
_class
loc:@Variable_3*
dtype0
~
Variable_3/Adam
VariableV2*
shape:x*
dtype0*
	container *
_class
loc:@Variable_3*
shared_name 
�
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
Y
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3
s
#Variable_3/Adam_1/Initializer/zerosConst*
valueBx*    *
_class
loc:@Variable_3*
dtype0
�
Variable_3/Adam_1
VariableV2*
shape:x*
dtype0*
	container *
_class
loc:@Variable_3*
shared_name 
�
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
]
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3
v
!Variable_4/Adam/Initializer/zerosConst*
valueB	� d*    *
_class
loc:@Variable_4*
dtype0
�
Variable_4/Adam
VariableV2*
shape:	� d*
dtype0*
	container *
_class
loc:@Variable_4*
shared_name 
�
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
Y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4
x
#Variable_4/Adam_1/Initializer/zerosConst*
valueB	� d*    *
_class
loc:@Variable_4*
dtype0
�
Variable_4/Adam_1
VariableV2*
shape:	� d*
dtype0*
	container *
_class
loc:@Variable_4*
shared_name 
�
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
]
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4
q
!Variable_5/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_5*
dtype0
~
Variable_5/Adam
VariableV2*
shape:d*
dtype0*
	container *
_class
loc:@Variable_5*
shared_name 
�
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
Y
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5
s
#Variable_5/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_5*
dtype0
�
Variable_5/Adam_1
VariableV2*
shape:d*
dtype0*
	container *
_class
loc:@Variable_5*
shared_name 
�
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
]
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5
u
!Variable_6/Adam/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_6*
dtype0
�
Variable_6/Adam
VariableV2*
shape
:d*
dtype0*
	container *
_class
loc:@Variable_6*
shared_name 
�
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
Y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6
w
#Variable_6/Adam_1/Initializer/zerosConst*
valueBd*    *
_class
loc:@Variable_6*
dtype0
�
Variable_6/Adam_1
VariableV2*
shape
:d*
dtype0*
	container *
_class
loc:@Variable_6*
shared_name 
�
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
]
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6
q
!Variable_7/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
dtype0
~
Variable_7/Adam
VariableV2*
shape:*
dtype0*
	container *
_class
loc:@Variable_7*
shared_name 
�
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
Y
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7
s
#Variable_7/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@Variable_7*
dtype0
�
Variable_7/Adam_1
VariableV2*
shape:*
dtype0*
	container *
_class
loc:@Variable_7*
shared_name 
�
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
]
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7
?
Adam/learning_rateConst*
valueB
 *��8*
dtype0
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w�?*
dtype0
9
Adam/epsilonConst*
valueB
 *w�+2*
dtype0
�
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/depthwise_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable*
use_locking( 
�
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_1*
use_locking( 
�
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon5gradients/depthwise_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_2*
use_locking( 
�
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_3*
use_locking( 
�
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_4*
use_locking( 
�
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_5*
use_locking( 
�
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_6*
use_locking( 
�
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
use_nesterov( *
_class
loc:@Variable_7*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable
{
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking( 
�
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
:
ArgMax/dimensionConst*
value	B :*
dtype0
Y
ArgMaxArgMaxlabels_outputArgMax/dimension*
T0*

Tidx0*
output_type0	
<
ArgMax_1/dimensionConst*
value	B :*
dtype0
[
ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*
T0*

Tidx0*
output_type0	
)
EqualEqualArgMaxArgMax_1*
T0	
+
CastCastEqual*

DstT0*

SrcT0

5
Const_5Const*
valueB: *
dtype0
A
MeanMeanCastConst_5*
T0*

Tidx0*
	keep_dims( 
8

save/ConstConst*
valueB Bmodel*
dtype0
�
save/SaveV2/tensor_namesConst*�
value�B�BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power*
dtype0
{
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_powerbeta2_power*(
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
P
save/RestoreV2/tensor_namesConst*
valueBBVariable*
dtype0
L
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
~
save/AssignAssignVariablesave/RestoreV2*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
W
save/RestoreV2_1/tensor_namesConst*"
valueBBVariable/Adam*
dtype0
N
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2
�
save/Assign_1AssignVariable/Adamsave/RestoreV2_1*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
Y
save/RestoreV2_2/tensor_namesConst*$
valueBBVariable/Adam_1*
dtype0
N
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2
�
save/Assign_2AssignVariable/Adam_1save/RestoreV2_2*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
T
save/RestoreV2_3/tensor_namesConst*
valueBB
Variable_1*
dtype0
N
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2
�
save/Assign_3Assign
Variable_1save/RestoreV2_3*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
Y
save/RestoreV2_4/tensor_namesConst*$
valueBBVariable_1/Adam*
dtype0
N
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2
�
save/Assign_4AssignVariable_1/Adamsave/RestoreV2_4*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
[
save/RestoreV2_5/tensor_namesConst*&
valueBBVariable_1/Adam_1*
dtype0
N
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2
�
save/Assign_5AssignVariable_1/Adam_1save/RestoreV2_5*
T0*
validate_shape(*
_class
loc:@Variable_1*
use_locking(
T
save/RestoreV2_6/tensor_namesConst*
valueBB
Variable_2*
dtype0
N
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2
�
save/Assign_6Assign
Variable_2save/RestoreV2_6*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
Y
save/RestoreV2_7/tensor_namesConst*$
valueBBVariable_2/Adam*
dtype0
N
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2
�
save/Assign_7AssignVariable_2/Adamsave/RestoreV2_7*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
[
save/RestoreV2_8/tensor_namesConst*&
valueBBVariable_2/Adam_1*
dtype0
N
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2
�
save/Assign_8AssignVariable_2/Adam_1save/RestoreV2_8*
T0*
validate_shape(*
_class
loc:@Variable_2*
use_locking(
T
save/RestoreV2_9/tensor_namesConst*
valueBB
Variable_3*
dtype0
N
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2
�
save/Assign_9Assign
Variable_3save/RestoreV2_9*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
Z
save/RestoreV2_10/tensor_namesConst*$
valueBBVariable_3/Adam*
dtype0
O
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2
�
save/Assign_10AssignVariable_3/Adamsave/RestoreV2_10*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
\
save/RestoreV2_11/tensor_namesConst*&
valueBBVariable_3/Adam_1*
dtype0
O
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2
�
save/Assign_11AssignVariable_3/Adam_1save/RestoreV2_11*
T0*
validate_shape(*
_class
loc:@Variable_3*
use_locking(
U
save/RestoreV2_12/tensor_namesConst*
valueBB
Variable_4*
dtype0
O
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2
�
save/Assign_12Assign
Variable_4save/RestoreV2_12*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
Z
save/RestoreV2_13/tensor_namesConst*$
valueBBVariable_4/Adam*
dtype0
O
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2
�
save/Assign_13AssignVariable_4/Adamsave/RestoreV2_13*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
\
save/RestoreV2_14/tensor_namesConst*&
valueBBVariable_4/Adam_1*
dtype0
O
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2
�
save/Assign_14AssignVariable_4/Adam_1save/RestoreV2_14*
T0*
validate_shape(*
_class
loc:@Variable_4*
use_locking(
U
save/RestoreV2_15/tensor_namesConst*
valueBB
Variable_5*
dtype0
O
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2
�
save/Assign_15Assign
Variable_5save/RestoreV2_15*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
Z
save/RestoreV2_16/tensor_namesConst*$
valueBBVariable_5/Adam*
dtype0
O
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2
�
save/Assign_16AssignVariable_5/Adamsave/RestoreV2_16*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
\
save/RestoreV2_17/tensor_namesConst*&
valueBBVariable_5/Adam_1*
dtype0
O
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2
�
save/Assign_17AssignVariable_5/Adam_1save/RestoreV2_17*
T0*
validate_shape(*
_class
loc:@Variable_5*
use_locking(
U
save/RestoreV2_18/tensor_namesConst*
valueBB
Variable_6*
dtype0
O
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2
�
save/Assign_18Assign
Variable_6save/RestoreV2_18*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
Z
save/RestoreV2_19/tensor_namesConst*$
valueBBVariable_6/Adam*
dtype0
O
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2
�
save/Assign_19AssignVariable_6/Adamsave/RestoreV2_19*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
\
save/RestoreV2_20/tensor_namesConst*&
valueBBVariable_6/Adam_1*
dtype0
O
"save/RestoreV2_20/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2
�
save/Assign_20AssignVariable_6/Adam_1save/RestoreV2_20*
T0*
validate_shape(*
_class
loc:@Variable_6*
use_locking(
U
save/RestoreV2_21/tensor_namesConst*
valueBB
Variable_7*
dtype0
O
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2
�
save/Assign_21Assign
Variable_7save/RestoreV2_21*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
Z
save/RestoreV2_22/tensor_namesConst*$
valueBBVariable_7/Adam*
dtype0
O
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2
�
save/Assign_22AssignVariable_7/Adamsave/RestoreV2_22*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
\
save/RestoreV2_23/tensor_namesConst*&
valueBBVariable_7/Adam_1*
dtype0
O
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2
�
save/Assign_23AssignVariable_7/Adam_1save/RestoreV2_23*
T0*
validate_shape(*
_class
loc:@Variable_7*
use_locking(
V
save/RestoreV2_24/tensor_namesConst* 
valueBBbeta1_power*
dtype0
O
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2
�
save/Assign_24Assignbeta1_powersave/RestoreV2_24*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
V
save/RestoreV2_25/tensor_namesConst* 
valueBBbeta2_power*
dtype0
O
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2
�
save/Assign_25Assignbeta2_powersave/RestoreV2_25*
T0*
validate_shape(*
_class
loc:@Variable*
use_locking(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25
�
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign"