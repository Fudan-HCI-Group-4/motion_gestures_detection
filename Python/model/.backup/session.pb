
F
x_inputPlaceholder*
dtype0*!
shape:ĸĸĸĸĸĸĸĸĸ
J
Reshape/shapeConst*
dtype0*%
valueB"ĸĸĸĸ         
A
ReshapeReshapex_inputReshape/shape*
T0*
Tshape0
E
PlaceholderPlaceholder*
dtype0*
shape:ĸĸĸĸĸĸĸĸĸ
S
truncated_normal/shapeConst*
dtype0*%
valueB"            
B
truncated_normal/meanConst*
dtype0*
valueB
 *    
D
truncated_normal/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
z
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
dtype0*

seed *
seed2 *
T0
_
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0
M
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0
d
Variable
VariableV2*
shape:*
dtype0*
shared_name *
	container 

Variable/AssignAssignVariabletruncated_normal*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
I
Variable/readIdentityVariable*
T0*
_class
loc:@Variable
6
ConstConst*
dtype0*
valueB(*    
Z

Variable_1
VariableV2*
shape:(*
dtype0*
shared_name *
	container 

Variable_1/AssignAssign
Variable_1Const*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
O
Variable_1/readIdentity
Variable_1*
T0*
_class
loc:@Variable_1
L
depthwise/ShapeConst*
dtype0*%
valueB"            
L
depthwise/dilation_rateConst*
dtype0*
valueB"      

	depthwiseDepthwiseConv2dNativeReshapeVariable/read*
T0*
strides
*
paddingVALID*
data_formatNHWC
/
AddAdd	depthwiseVariable_1/read*
T0

ReluReluAdd*
T0
t
MaxPoolMaxPoolRelu*
T0*
strides
*
paddingVALID*
ksize
*
data_formatNHWC
U
truncated_normal_1/shapeConst*
dtype0*%
valueB"      (      
D
truncated_normal_1/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_1/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_1/TruncatedNormalTruncatedNormaltruncated_normal_1/shape*
dtype0*

seed *
seed2 *
T0
e
truncated_normal_1/mulMul"truncated_normal_1/TruncatedNormaltruncated_normal_1/stddev*
T0
S
truncated_normal_1Addtruncated_normal_1/multruncated_normal_1/mean*
T0
f

Variable_2
VariableV2*
shape:(*
dtype0*
shared_name *
	container 

Variable_2/AssignAssign
Variable_2truncated_normal_1*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
O
Variable_2/readIdentity
Variable_2*
T0*
_class
loc:@Variable_2
8
Const_1Const*
dtype0*
valueBP*    
Z

Variable_3
VariableV2*
shape:P*
dtype0*
shared_name *
	container 

Variable_3/AssignAssign
Variable_3Const_1*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
O
Variable_3/readIdentity
Variable_3*
T0*
_class
loc:@Variable_3
N
depthwise_1/ShapeConst*
dtype0*%
valueB"      (      
N
depthwise_1/dilation_rateConst*
dtype0*
valueB"      

depthwise_1DepthwiseConv2dNativeMaxPoolVariable_2/read*
T0*
strides
*
paddingVALID*
data_formatNHWC
3
Add_1Adddepthwise_1Variable_3/read*
T0

Relu_1ReluAdd_1*
T0
D
Reshape_1/shapeConst*
dtype0*
valueB"ĸĸĸĸð
  
D
	Reshape_1ReshapeRelu_1Reshape_1/shape*
T0*
Tshape0
M
truncated_normal_2/shapeConst*
dtype0*
valueB"ð
  d   
D
truncated_normal_2/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_2/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_2/TruncatedNormalTruncatedNormaltruncated_normal_2/shape*
dtype0*

seed *
seed2 *
T0
e
truncated_normal_2/mulMul"truncated_normal_2/TruncatedNormaltruncated_normal_2/stddev*
T0
S
truncated_normal_2Addtruncated_normal_2/multruncated_normal_2/mean*
T0
_

Variable_4
VariableV2*
shape:	ðd*
dtype0*
shared_name *
	container 

Variable_4/AssignAssign
Variable_4truncated_normal_2*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
O
Variable_4/readIdentity
Variable_4*
T0*
_class
loc:@Variable_4
8
Const_2Const*
dtype0*
valueBd*    
Z

Variable_5
VariableV2*
shape:d*
dtype0*
shared_name *
	container 

Variable_5/AssignAssign
Variable_5Const_2*
T0*
_class
loc:@Variable_5*
validate_shape(*
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
transpose_a( *
transpose_b( 
.
Add_2AddMatMulVariable_5/read*
T0

TanhTanhAdd_2*
T0
M
truncated_normal_3/shapeConst*
dtype0*
valueB"d      
D
truncated_normal_3/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_3/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_3/TruncatedNormalTruncatedNormaltruncated_normal_3/shape*
dtype0*

seed *
seed2 *
T0
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
:d*
dtype0*
shared_name *
	container 

Variable_6/AssignAssign
Variable_6truncated_normal_3*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
O
Variable_6/readIdentity
Variable_6*
T0*
_class
loc:@Variable_6
8
Const_3Const*
dtype0*
valueB*    
Z

Variable_7
VariableV2*
shape:*
dtype0*
shared_name *
	container 

Variable_7/AssignAssign
Variable_7Const_3*
T0*
_class
loc:@Variable_7*
validate_shape(*
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
transpose_a( *
transpose_b( 
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
dtype0*
valueB"       
>
SumSummulConst_4*
T0*
	keep_dims( *

Tidx0

NegNegSum*
T0
8
gradients/ShapeConst*
dtype0*
valueB 
<
gradients/ConstConst*
dtype0*
valueB
 *  ?
A
gradients/FillFillgradients/Shapegradients/Const*
T0
6
gradients/Neg_grad/NegNeggradients/Fill*
T0
U
 gradients/Sum_grad/Reshape/shapeConst*
dtype0*
valueB"      
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

(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0
D
gradients/mul_grad/mulMulgradients/Sum_grad/TileLog*
T0

gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
n
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0
N
gradients/mul_grad/mul_1MulPlaceholdergradients/Sum_grad/Tile*
T0

gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
t
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
ą
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_grad/Reshape
·
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
2gradients/labels_output_grad/Sum/reduction_indicesConst*
dtype0*
valueB:
Ģ
 gradients/labels_output_grad/SumSum gradients/labels_output_grad/mul2gradients/labels_output_grad/Sum/reduction_indices*
T0*
	keep_dims( *

Tidx0
_
*gradients/labels_output_grad/Reshape/shapeConst*
dtype0*
valueB"ĸĸĸĸ   

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
gradients/add_grad/Shape_1Const*
dtype0*
valueB:

(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0

gradients/add_grad/SumSum"gradients/labels_output_grad/mul_1(gradients/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
n
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0

gradients/add_grad/Sum_1Sum"gradients/labels_output_grad/mul_1*gradients/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
t
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
ą
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
·
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1

gradients/MatMul_1_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyVariable_6/read*
T0*
transpose_a( *
transpose_b(

 gradients/MatMul_1_grad/MatMul_1MatMulTanh+gradients/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
t
(gradients/MatMul_1_grad/tuple/group_depsNoOp^gradients/MatMul_1_grad/MatMul!^gradients/MatMul_1_grad/MatMul_1
Ã
0gradients/MatMul_1_grad/tuple/control_dependencyIdentitygradients/MatMul_1_grad/MatMul)^gradients/MatMul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_1_grad/MatMul
É
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
gradients/Add_2_grad/Shape_1Const*
dtype0*
valueB:d

*gradients/Add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_2_grad/Shapegradients/Add_2_grad/Shape_1*
T0

gradients/Add_2_grad/SumSumgradients/Tanh_grad/TanhGrad*gradients/Add_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
t
gradients/Add_2_grad/ReshapeReshapegradients/Add_2_grad/Sumgradients/Add_2_grad/Shape*
T0*
Tshape0

gradients/Add_2_grad/Sum_1Sumgradients/Tanh_grad/TanhGrad,gradients/Add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
z
gradients/Add_2_grad/Reshape_1Reshapegradients/Add_2_grad/Sum_1gradients/Add_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_2_grad/tuple/group_depsNoOp^gradients/Add_2_grad/Reshape^gradients/Add_2_grad/Reshape_1
đ
-gradients/Add_2_grad/tuple/control_dependencyIdentitygradients/Add_2_grad/Reshape&^gradients/Add_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_2_grad/Reshape
ŋ
/gradients/Add_2_grad/tuple/control_dependency_1Identitygradients/Add_2_grad/Reshape_1&^gradients/Add_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_2_grad/Reshape_1

gradients/MatMul_grad/MatMulMatMul-gradients/Add_2_grad/tuple/control_dependencyVariable_4/read*
T0*
transpose_a( *
transpose_b(

gradients/MatMul_grad/MatMul_1MatMul	Reshape_1-gradients/Add_2_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ŧ
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
Á
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
H
gradients/Reshape_1_grad/ShapeShapeRelu_1*
T0*
out_type0

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
gradients/Add_1_grad/Shape_1Const*
dtype0*
valueB:P

*gradients/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_1_grad/Shapegradients/Add_1_grad/Shape_1*
T0

gradients/Add_1_grad/SumSumgradients/Relu_1_grad/ReluGrad*gradients/Add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
t
gradients/Add_1_grad/ReshapeReshapegradients/Add_1_grad/Sumgradients/Add_1_grad/Shape*
T0*
Tshape0

gradients/Add_1_grad/Sum_1Sumgradients/Relu_1_grad/ReluGrad,gradients/Add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
z
gradients/Add_1_grad/Reshape_1Reshapegradients/Add_1_grad/Sum_1gradients/Add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/Add_1_grad/tuple/group_depsNoOp^gradients/Add_1_grad/Reshape^gradients/Add_1_grad/Reshape_1
đ
-gradients/Add_1_grad/tuple/control_dependencyIdentitygradients/Add_1_grad/Reshape&^gradients/Add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_1_grad/Reshape
ŋ
/gradients/Add_1_grad/tuple/control_dependency_1Identitygradients/Add_1_grad/Reshape_1&^gradients/Add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Add_1_grad/Reshape_1
K
 gradients/depthwise_1_grad/ShapeShapeMaxPool*
T0*
out_type0

=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput gradients/depthwise_1_grad/ShapeVariable_2/read-gradients/Add_1_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
_
"gradients/depthwise_1_grad/Shape_1Const*
dtype0*%
valueB"      (      

>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterMaxPool"gradients/depthwise_1_grad/Shape_1-gradients/Add_1_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
ī
+gradients/depthwise_1_grad/tuple/group_depsNoOp>^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput?^gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter

3gradients/depthwise_1_grad/tuple/control_dependencyIdentity=gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput,^gradients/depthwise_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropInput

5gradients/depthwise_1_grad/tuple/control_dependency_1Identity>gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter,^gradients/depthwise_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/depthwise_1_grad/DepthwiseConv2dNativeBackpropFilter
Ņ
"gradients/MaxPool_grad/MaxPoolGradMaxPoolGradReluMaxPool3gradients/depthwise_1_grad/tuple/control_dependency*
strides
*
paddingVALID*
data_formatNHWC*
ksize
*
T0
[
gradients/Relu_grad/ReluGradReluGrad"gradients/MaxPool_grad/MaxPoolGradRelu*
T0
E
gradients/Add_grad/ShapeShape	depthwise*
T0*
out_type0
H
gradients/Add_grad/Shape_1Const*
dtype0*
valueB:(

(gradients/Add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Add_grad/Shapegradients/Add_grad/Shape_1*
T0

gradients/Add_grad/SumSumgradients/Relu_grad/ReluGrad(gradients/Add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
n
gradients/Add_grad/ReshapeReshapegradients/Add_grad/Sumgradients/Add_grad/Shape*
T0*
Tshape0

gradients/Add_grad/Sum_1Sumgradients/Relu_grad/ReluGrad*gradients/Add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0
t
gradients/Add_grad/Reshape_1Reshapegradients/Add_grad/Sum_1gradients/Add_grad/Shape_1*
T0*
Tshape0
g
#gradients/Add_grad/tuple/group_depsNoOp^gradients/Add_grad/Reshape^gradients/Add_grad/Reshape_1
ą
+gradients/Add_grad/tuple/control_dependencyIdentitygradients/Add_grad/Reshape$^gradients/Add_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Add_grad/Reshape
·
-gradients/Add_grad/tuple/control_dependency_1Identitygradients/Add_grad/Reshape_1$^gradients/Add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Add_grad/Reshape_1
I
gradients/depthwise_grad/ShapeShapeReshape*
T0*
out_type0

;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInputgradients/depthwise_grad/ShapeVariable/read+gradients/Add_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
]
 gradients/depthwise_grad/Shape_1Const*
dtype0*%
valueB"            

<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterReshape gradients/depthwise_grad/Shape_1+gradients/Add_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
Ū
)gradients/depthwise_grad/tuple/group_depsNoOp<^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput=^gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
ĸ
1gradients/depthwise_grad/tuple/control_dependencyIdentity;gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput*^gradients/depthwise_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput

3gradients/depthwise_grad/tuple/control_dependency_1Identity<gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter*^gradients/depthwise_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter
c
beta1_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *fff?
t
beta1_power
VariableV2*
shape: *
dtype0*
_class
loc:@Variable*
shared_name *
	container 

beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
O
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@Variable
c
beta2_power/initial_valueConst*
dtype0*
_class
loc:@Variable*
valueB
 *wū?
t
beta2_power
VariableV2*
shape: *
dtype0*
_class
loc:@Variable*
shared_name *
	container 

beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
O
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@Variable
y
Variable/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable*%
valueB*    

Variable/Adam
VariableV2*
shape:*
dtype0*
_class
loc:@Variable*
shared_name *
	container 

Variable/Adam/AssignAssignVariable/AdamVariable/Adam/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
S
Variable/Adam/readIdentityVariable/Adam*
T0*
_class
loc:@Variable
{
!Variable/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable*%
valueB*    

Variable/Adam_1
VariableV2*
shape:*
dtype0*
_class
loc:@Variable*
shared_name *
	container 
Ģ
Variable/Adam_1/AssignAssignVariable/Adam_1!Variable/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
W
Variable/Adam_1/readIdentityVariable/Adam_1*
T0*
_class
loc:@Variable
q
!Variable_1/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_1*
valueB(*    
~
Variable_1/Adam
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_1*
shared_name *
	container 
Ĩ
Variable_1/Adam/AssignAssignVariable_1/Adam!Variable_1/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
Y
Variable_1/Adam/readIdentityVariable_1/Adam*
T0*
_class
loc:@Variable_1
s
#Variable_1/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_1*
valueB(*    

Variable_1/Adam_1
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_1*
shared_name *
	container 
Ŧ
Variable_1/Adam_1/AssignAssignVariable_1/Adam_1#Variable_1/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
]
Variable_1/Adam_1/readIdentityVariable_1/Adam_1*
T0*
_class
loc:@Variable_1
}
!Variable_2/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_2*%
valueB(*    

Variable_2/Adam
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_2*
shared_name *
	container 
Ĩ
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
Y
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_class
loc:@Variable_2

#Variable_2/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_2*%
valueB(*    

Variable_2/Adam_1
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_2*
shared_name *
	container 
Ŧ
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
]
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_class
loc:@Variable_2
q
!Variable_3/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_3*
valueBP*    
~
Variable_3/Adam
VariableV2*
shape:P*
dtype0*
_class
loc:@Variable_3*
shared_name *
	container 
Ĩ
Variable_3/Adam/AssignAssignVariable_3/Adam!Variable_3/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
Y
Variable_3/Adam/readIdentityVariable_3/Adam*
T0*
_class
loc:@Variable_3
s
#Variable_3/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_3*
valueBP*    

Variable_3/Adam_1
VariableV2*
shape:P*
dtype0*
_class
loc:@Variable_3*
shared_name *
	container 
Ŧ
Variable_3/Adam_1/AssignAssignVariable_3/Adam_1#Variable_3/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
]
Variable_3/Adam_1/readIdentityVariable_3/Adam_1*
T0*
_class
loc:@Variable_3
v
!Variable_4/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_4*
valueB	ðd*    

Variable_4/Adam
VariableV2*
shape:	ðd*
dtype0*
_class
loc:@Variable_4*
shared_name *
	container 
Ĩ
Variable_4/Adam/AssignAssignVariable_4/Adam!Variable_4/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
Y
Variable_4/Adam/readIdentityVariable_4/Adam*
T0*
_class
loc:@Variable_4
x
#Variable_4/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_4*
valueB	ðd*    

Variable_4/Adam_1
VariableV2*
shape:	ðd*
dtype0*
_class
loc:@Variable_4*
shared_name *
	container 
Ŧ
Variable_4/Adam_1/AssignAssignVariable_4/Adam_1#Variable_4/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
]
Variable_4/Adam_1/readIdentityVariable_4/Adam_1*
T0*
_class
loc:@Variable_4
q
!Variable_5/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_5*
valueBd*    
~
Variable_5/Adam
VariableV2*
shape:d*
dtype0*
_class
loc:@Variable_5*
shared_name *
	container 
Ĩ
Variable_5/Adam/AssignAssignVariable_5/Adam!Variable_5/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
Y
Variable_5/Adam/readIdentityVariable_5/Adam*
T0*
_class
loc:@Variable_5
s
#Variable_5/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_5*
valueBd*    

Variable_5/Adam_1
VariableV2*
shape:d*
dtype0*
_class
loc:@Variable_5*
shared_name *
	container 
Ŧ
Variable_5/Adam_1/AssignAssignVariable_5/Adam_1#Variable_5/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
]
Variable_5/Adam_1/readIdentityVariable_5/Adam_1*
T0*
_class
loc:@Variable_5
u
!Variable_6/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_6*
valueBd*    

Variable_6/Adam
VariableV2*
shape
:d*
dtype0*
_class
loc:@Variable_6*
shared_name *
	container 
Ĩ
Variable_6/Adam/AssignAssignVariable_6/Adam!Variable_6/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
Y
Variable_6/Adam/readIdentityVariable_6/Adam*
T0*
_class
loc:@Variable_6
w
#Variable_6/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_6*
valueBd*    

Variable_6/Adam_1
VariableV2*
shape
:d*
dtype0*
_class
loc:@Variable_6*
shared_name *
	container 
Ŧ
Variable_6/Adam_1/AssignAssignVariable_6/Adam_1#Variable_6/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
]
Variable_6/Adam_1/readIdentityVariable_6/Adam_1*
T0*
_class
loc:@Variable_6
q
!Variable_7/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_7*
valueB*    
~
Variable_7/Adam
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_7*
shared_name *
	container 
Ĩ
Variable_7/Adam/AssignAssignVariable_7/Adam!Variable_7/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
Y
Variable_7/Adam/readIdentityVariable_7/Adam*
T0*
_class
loc:@Variable_7
s
#Variable_7/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_7*
valueB*    

Variable_7/Adam_1
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_7*
shared_name *
	container 
Ŧ
Variable_7/Adam_1/AssignAssignVariable_7/Adam_1#Variable_7/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
]
Variable_7/Adam_1/readIdentityVariable_7/Adam_1*
T0*
_class
loc:@Variable_7
?
Adam/learning_rateConst*
dtype0*
valueB
 *·Ņ8
7

Adam/beta1Const*
dtype0*
valueB
 *fff?
7

Adam/beta2Const*
dtype0*
valueB
 *wū?
9
Adam/epsilonConst*
dtype0*
valueB
 *wĖ+2
ĩ
Adam/update_Variable/ApplyAdam	ApplyAdamVariableVariable/AdamVariable/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon3gradients/depthwise_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable*
use_nesterov( *
use_locking( 
đ
 Adam/update_Variable_1/ApplyAdam	ApplyAdam
Variable_1Variable_1/AdamVariable_1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/Add_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_1*
use_nesterov( *
use_locking( 
Á
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon5gradients/depthwise_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_2*
use_nesterov( *
use_locking( 
ŧ
 Adam/update_Variable_3/ApplyAdam	ApplyAdam
Variable_3Variable_3/AdamVariable_3/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_3*
use_nesterov( *
use_locking( 
ž
 Adam/update_Variable_4/ApplyAdam	ApplyAdam
Variable_4Variable_4/AdamVariable_4/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_4*
use_nesterov( *
use_locking( 
ŧ
 Adam/update_Variable_5/ApplyAdam	ApplyAdam
Variable_5Variable_5/AdamVariable_5/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon/gradients/Add_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_5*
use_nesterov( *
use_locking( 
ū
 Adam/update_Variable_6/ApplyAdam	ApplyAdam
Variable_6Variable_6/AdamVariable_6/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon2gradients/MatMul_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_6*
use_nesterov( *
use_locking( 
đ
 Adam/update_Variable_7/ApplyAdam	ApplyAdam
Variable_7Variable_7/AdamVariable_7/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_7*
use_nesterov( *
use_locking( 
é
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable
{
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking( 
ë

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam*
T0*
_class
loc:@Variable

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking( 
Ā
AdamNoOp^Adam/update_Variable/ApplyAdam!^Adam/update_Variable_1/ApplyAdam!^Adam/update_Variable_2/ApplyAdam!^Adam/update_Variable_3/ApplyAdam!^Adam/update_Variable_4/ApplyAdam!^Adam/update_Variable_5/ApplyAdam!^Adam/update_Variable_6/ApplyAdam!^Adam/update_Variable_7/ApplyAdam^Adam/Assign^Adam/Assign_1
:
ArgMax/dimensionConst*
dtype0*
value	B :
Y
ArgMaxArgMaxlabels_outputArgMax/dimension*
T0*
output_type0	*

Tidx0
<
ArgMax_1/dimensionConst*
dtype0*
value	B :
[
ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*
T0*
output_type0	*

Tidx0
)
EqualEqualArgMaxArgMax_1*
T0	
+
CastCastEqual*

SrcT0
*

DstT0
5
Const_5Const*
dtype0*
valueB: 
A
MeanMeanCastConst_5*
T0*
	keep_dims( *

Tidx0
8

save/ConstConst*
dtype0*
valueB Bmodel
Ú
save/SaveV2/tensor_namesConst*
dtype0*Đ
valueBBVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1Bbeta1_powerBbeta2_power
{
save/SaveV2/shape_and_slicesConst*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 

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
dtype0*
valueBBVariable
L
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B 
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2
~
save/AssignAssignVariablesave/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
W
save/RestoreV2_1/tensor_namesConst*
dtype0*"
valueBBVariable/Adam
N
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2

save/Assign_1AssignVariable/Adamsave/RestoreV2_1*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
Y
save/RestoreV2_2/tensor_namesConst*
dtype0*$
valueBBVariable/Adam_1
N
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2

save/Assign_2AssignVariable/Adam_1save/RestoreV2_2*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
T
save/RestoreV2_3/tensor_namesConst*
dtype0*
valueBB
Variable_1
N
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2

save/Assign_3Assign
Variable_1save/RestoreV2_3*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
Y
save/RestoreV2_4/tensor_namesConst*
dtype0*$
valueBBVariable_1/Adam
N
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2

save/Assign_4AssignVariable_1/Adamsave/RestoreV2_4*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
[
save/RestoreV2_5/tensor_namesConst*
dtype0*&
valueBBVariable_1/Adam_1
N
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2

save/Assign_5AssignVariable_1/Adam_1save/RestoreV2_5*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
T
save/RestoreV2_6/tensor_namesConst*
dtype0*
valueBB
Variable_2
N
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2

save/Assign_6Assign
Variable_2save/RestoreV2_6*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
Y
save/RestoreV2_7/tensor_namesConst*
dtype0*$
valueBBVariable_2/Adam
N
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2

save/Assign_7AssignVariable_2/Adamsave/RestoreV2_7*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
[
save/RestoreV2_8/tensor_namesConst*
dtype0*&
valueBBVariable_2/Adam_1
N
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2

save/Assign_8AssignVariable_2/Adam_1save/RestoreV2_8*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
T
save/RestoreV2_9/tensor_namesConst*
dtype0*
valueBB
Variable_3
N
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2

save/Assign_9Assign
Variable_3save/RestoreV2_9*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
Z
save/RestoreV2_10/tensor_namesConst*
dtype0*$
valueBBVariable_3/Adam
O
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2

save/Assign_10AssignVariable_3/Adamsave/RestoreV2_10*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
\
save/RestoreV2_11/tensor_namesConst*
dtype0*&
valueBBVariable_3/Adam_1
O
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2

save/Assign_11AssignVariable_3/Adam_1save/RestoreV2_11*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
U
save/RestoreV2_12/tensor_namesConst*
dtype0*
valueBB
Variable_4
O
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2

save/Assign_12Assign
Variable_4save/RestoreV2_12*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
Z
save/RestoreV2_13/tensor_namesConst*
dtype0*$
valueBBVariable_4/Adam
O
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2

save/Assign_13AssignVariable_4/Adamsave/RestoreV2_13*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
\
save/RestoreV2_14/tensor_namesConst*
dtype0*&
valueBBVariable_4/Adam_1
O
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2

save/Assign_14AssignVariable_4/Adam_1save/RestoreV2_14*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
U
save/RestoreV2_15/tensor_namesConst*
dtype0*
valueBB
Variable_5
O
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2

save/Assign_15Assign
Variable_5save/RestoreV2_15*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
Z
save/RestoreV2_16/tensor_namesConst*
dtype0*$
valueBBVariable_5/Adam
O
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2

save/Assign_16AssignVariable_5/Adamsave/RestoreV2_16*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
\
save/RestoreV2_17/tensor_namesConst*
dtype0*&
valueBBVariable_5/Adam_1
O
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2

save/Assign_17AssignVariable_5/Adam_1save/RestoreV2_17*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
U
save/RestoreV2_18/tensor_namesConst*
dtype0*
valueBB
Variable_6
O
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2

save/Assign_18Assign
Variable_6save/RestoreV2_18*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
Z
save/RestoreV2_19/tensor_namesConst*
dtype0*$
valueBBVariable_6/Adam
O
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2

save/Assign_19AssignVariable_6/Adamsave/RestoreV2_19*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
\
save/RestoreV2_20/tensor_namesConst*
dtype0*&
valueBBVariable_6/Adam_1
O
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2

save/Assign_20AssignVariable_6/Adam_1save/RestoreV2_20*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
U
save/RestoreV2_21/tensor_namesConst*
dtype0*
valueBB
Variable_7
O
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2

save/Assign_21Assign
Variable_7save/RestoreV2_21*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
Z
save/RestoreV2_22/tensor_namesConst*
dtype0*$
valueBBVariable_7/Adam
O
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2

save/Assign_22AssignVariable_7/Adamsave/RestoreV2_22*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
\
save/RestoreV2_23/tensor_namesConst*
dtype0*&
valueBBVariable_7/Adam_1
O
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2

save/Assign_23AssignVariable_7/Adam_1save/RestoreV2_23*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
V
save/RestoreV2_24/tensor_namesConst*
dtype0* 
valueBBbeta1_power
O
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2

save/Assign_24Assignbeta1_powersave/RestoreV2_24*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
V
save/RestoreV2_25/tensor_namesConst*
dtype0* 
valueBBbeta2_power
O
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2

save/Assign_25Assignbeta2_powersave/RestoreV2_25*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
Æ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25
ð
initNoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign
H
	x_input_1Placeholder*
dtype0*!
shape:ĸĸĸĸĸĸĸĸĸ
L
Reshape_2/shapeConst*
dtype0*%
valueB"ĸĸĸĸ         
G
	Reshape_2Reshape	x_input_1Reshape_2/shape*
T0*
Tshape0
G
Placeholder_1Placeholder*
dtype0*
shape:ĸĸĸĸĸĸĸĸĸ
U
truncated_normal_4/shapeConst*
dtype0*%
valueB"            
D
truncated_normal_4/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_4/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_4/TruncatedNormalTruncatedNormaltruncated_normal_4/shape*
dtype0*

seed *
seed2 *
T0
e
truncated_normal_4/mulMul"truncated_normal_4/TruncatedNormaltruncated_normal_4/stddev*
T0
S
truncated_normal_4Addtruncated_normal_4/multruncated_normal_4/mean*
T0
f

Variable_8
VariableV2*
shape:*
dtype0*
shared_name *
	container 

Variable_8/AssignAssign
Variable_8truncated_normal_4*
T0*
_class
loc:@Variable_8*
validate_shape(*
use_locking(
O
Variable_8/readIdentity
Variable_8*
T0*
_class
loc:@Variable_8
8
Const_6Const*
dtype0*
valueB(*    
Z

Variable_9
VariableV2*
shape:(*
dtype0*
shared_name *
	container 

Variable_9/AssignAssign
Variable_9Const_6*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
O
Variable_9/readIdentity
Variable_9*
T0*
_class
loc:@Variable_9
N
depthwise_2/ShapeConst*
dtype0*%
valueB"            
N
depthwise_2/dilation_rateConst*
dtype0*
valueB"      

depthwise_2DepthwiseConv2dNative	Reshape_2Variable_8/read*
T0*
strides
*
paddingVALID*
data_formatNHWC
3
Add_3Adddepthwise_2Variable_9/read*
T0

Relu_2ReluAdd_3*
T0
x
	MaxPool_1MaxPoolRelu_2*
T0*
strides
*
paddingVALID*
ksize
*
data_formatNHWC
U
truncated_normal_5/shapeConst*
dtype0*%
valueB"      (      
D
truncated_normal_5/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_5/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_5/TruncatedNormalTruncatedNormaltruncated_normal_5/shape*
dtype0*

seed *
seed2 *
T0
e
truncated_normal_5/mulMul"truncated_normal_5/TruncatedNormaltruncated_normal_5/stddev*
T0
S
truncated_normal_5Addtruncated_normal_5/multruncated_normal_5/mean*
T0
g
Variable_10
VariableV2*
shape:(*
dtype0*
shared_name *
	container 

Variable_10/AssignAssignVariable_10truncated_normal_5*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
R
Variable_10/readIdentityVariable_10*
T0*
_class
loc:@Variable_10
8
Const_7Const*
dtype0*
valueBP*    
[
Variable_11
VariableV2*
shape:P*
dtype0*
shared_name *
	container 

Variable_11/AssignAssignVariable_11Const_7*
T0*
_class
loc:@Variable_11*
validate_shape(*
use_locking(
R
Variable_11/readIdentityVariable_11*
T0*
_class
loc:@Variable_11
N
depthwise_3/ShapeConst*
dtype0*%
valueB"      (      
N
depthwise_3/dilation_rateConst*
dtype0*
valueB"      

depthwise_3DepthwiseConv2dNative	MaxPool_1Variable_10/read*
T0*
strides
*
paddingVALID*
data_formatNHWC
4
Add_4Adddepthwise_3Variable_11/read*
T0

Relu_3ReluAdd_4*
T0
D
Reshape_3/shapeConst*
dtype0*
valueB"ĸĸĸĸð
  
D
	Reshape_3ReshapeRelu_3Reshape_3/shape*
T0*
Tshape0
M
truncated_normal_6/shapeConst*
dtype0*
valueB"ð
  d   
D
truncated_normal_6/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_6/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_6/TruncatedNormalTruncatedNormaltruncated_normal_6/shape*
dtype0*

seed *
seed2 *
T0
e
truncated_normal_6/mulMul"truncated_normal_6/TruncatedNormaltruncated_normal_6/stddev*
T0
S
truncated_normal_6Addtruncated_normal_6/multruncated_normal_6/mean*
T0
`
Variable_12
VariableV2*
shape:	ðd*
dtype0*
shared_name *
	container 

Variable_12/AssignAssignVariable_12truncated_normal_6*
T0*
_class
loc:@Variable_12*
validate_shape(*
use_locking(
R
Variable_12/readIdentityVariable_12*
T0*
_class
loc:@Variable_12
8
Const_8Const*
dtype0*
valueBd*    
[
Variable_13
VariableV2*
shape:d*
dtype0*
shared_name *
	container 

Variable_13/AssignAssignVariable_13Const_8*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
R
Variable_13/readIdentityVariable_13*
T0*
_class
loc:@Variable_13
^
MatMul_2MatMul	Reshape_3Variable_12/read*
T0*
transpose_a( *
transpose_b( 
1
Add_5AddMatMul_2Variable_13/read*
T0

Tanh_1TanhAdd_5*
T0
M
truncated_normal_7/shapeConst*
dtype0*
valueB"d      
D
truncated_normal_7/meanConst*
dtype0*
valueB
 *    
F
truncated_normal_7/stddevConst*
dtype0*
valueB
 *ÍĖĖ=
~
"truncated_normal_7/TruncatedNormalTruncatedNormaltruncated_normal_7/shape*
dtype0*

seed *
seed2 *
T0
e
truncated_normal_7/mulMul"truncated_normal_7/TruncatedNormaltruncated_normal_7/stddev*
T0
S
truncated_normal_7Addtruncated_normal_7/multruncated_normal_7/mean*
T0
_
Variable_14
VariableV2*
shape
:d*
dtype0*
shared_name *
	container 

Variable_14/AssignAssignVariable_14truncated_normal_7*
T0*
_class
loc:@Variable_14*
validate_shape(*
use_locking(
R
Variable_14/readIdentityVariable_14*
T0*
_class
loc:@Variable_14
8
Const_9Const*
dtype0*
valueB*    
[
Variable_15
VariableV2*
shape:*
dtype0*
shared_name *
	container 

Variable_15/AssignAssignVariable_15Const_9*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
R
Variable_15/readIdentityVariable_15*
T0*
_class
loc:@Variable_15
[
MatMul_3MatMulTanh_1Variable_14/read*
T0*
transpose_a( *
transpose_b( 
1
add_1AddMatMul_3Variable_15/read*
T0
*
labels_output_1Softmaxadd_1*
T0
&
Log_1Loglabels_output_1*
T0
+
mul_1MulPlaceholder_1Log_1*
T0
=
Const_10Const*
dtype0*
valueB"       
C
Sum_1Summul_1Const_10*
T0*
	keep_dims( *

Tidx0

Neg_1NegSum_1*
T0
:
gradients_1/ShapeConst*
dtype0*
valueB 
>
gradients_1/ConstConst*
dtype0*
valueB
 *  ?
G
gradients_1/FillFillgradients_1/Shapegradients_1/Const*
T0
<
gradients_1/Neg_1_grad/NegNeggradients_1/Fill*
T0
Y
$gradients_1/Sum_1_grad/Reshape/shapeConst*
dtype0*
valueB"      

gradients_1/Sum_1_grad/ReshapeReshapegradients_1/Neg_1_grad/Neg$gradients_1/Sum_1_grad/Reshape/shape*
T0*
Tshape0
E
gradients_1/Sum_1_grad/ShapeShapemul_1*
T0*
out_type0
|
gradients_1/Sum_1_grad/TileTilegradients_1/Sum_1_grad/Reshapegradients_1/Sum_1_grad/Shape*
T0*

Tmultiples0
M
gradients_1/mul_1_grad/ShapeShapePlaceholder_1*
T0*
out_type0
G
gradients_1/mul_1_grad/Shape_1ShapeLog_1*
T0*
out_type0

,gradients_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_1_grad/Shapegradients_1/mul_1_grad/Shape_1*
T0
N
gradients_1/mul_1_grad/mulMulgradients_1/Sum_1_grad/TileLog_1*
T0

gradients_1/mul_1_grad/SumSumgradients_1/mul_1_grad/mul,gradients_1/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
z
gradients_1/mul_1_grad/ReshapeReshapegradients_1/mul_1_grad/Sumgradients_1/mul_1_grad/Shape*
T0*
Tshape0
X
gradients_1/mul_1_grad/mul_1MulPlaceholder_1gradients_1/Sum_1_grad/Tile*
T0

gradients_1/mul_1_grad/Sum_1Sumgradients_1/mul_1_grad/mul_1.gradients_1/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0

 gradients_1/mul_1_grad/Reshape_1Reshapegradients_1/mul_1_grad/Sum_1gradients_1/mul_1_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/mul_1_grad/tuple/group_depsNoOp^gradients_1/mul_1_grad/Reshape!^gradients_1/mul_1_grad/Reshape_1
Á
/gradients_1/mul_1_grad/tuple/control_dependencyIdentitygradients_1/mul_1_grad/Reshape(^gradients_1/mul_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/mul_1_grad/Reshape
Į
1gradients_1/mul_1_grad/tuple/control_dependency_1Identity gradients_1/mul_1_grad/Reshape_1(^gradients_1/mul_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/mul_1_grad/Reshape_1
}
!gradients_1/Log_1_grad/Reciprocal
Reciprocallabels_output_12^gradients_1/mul_1_grad/tuple/control_dependency_1*
T0

gradients_1/Log_1_grad/mulMul1gradients_1/mul_1_grad/tuple/control_dependency_1!gradients_1/Log_1_grad/Reciprocal*
T0
a
$gradients_1/labels_output_1_grad/mulMulgradients_1/Log_1_grad/mullabels_output_1*
T0
d
6gradients_1/labels_output_1_grad/Sum/reduction_indicesConst*
dtype0*
valueB:
Ŋ
$gradients_1/labels_output_1_grad/SumSum$gradients_1/labels_output_1_grad/mul6gradients_1/labels_output_1_grad/Sum/reduction_indices*
T0*
	keep_dims( *

Tidx0
c
.gradients_1/labels_output_1_grad/Reshape/shapeConst*
dtype0*
valueB"ĸĸĸĸ   
 
(gradients_1/labels_output_1_grad/ReshapeReshape$gradients_1/labels_output_1_grad/Sum.gradients_1/labels_output_1_grad/Reshape/shape*
T0*
Tshape0
z
$gradients_1/labels_output_1_grad/subSubgradients_1/Log_1_grad/mul(gradients_1/labels_output_1_grad/Reshape*
T0
m
&gradients_1/labels_output_1_grad/mul_1Mul$gradients_1/labels_output_1_grad/sublabels_output_1*
T0
H
gradients_1/add_1_grad/ShapeShapeMatMul_3*
T0*
out_type0
L
gradients_1/add_1_grad/Shape_1Const*
dtype0*
valueB:

,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0

gradients_1/add_1_grad/SumSum&gradients_1/labels_output_1_grad/mul_1,gradients_1/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
z
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
T0*
Tshape0
Ą
gradients_1/add_1_grad/Sum_1Sum&gradients_1/labels_output_1_grad/mul_1.gradients_1/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0

 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
Á
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape
Į
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1

 gradients_1/MatMul_3_grad/MatMulMatMul/gradients_1/add_1_grad/tuple/control_dependencyVariable_14/read*
T0*
transpose_a( *
transpose_b(

"gradients_1/MatMul_3_grad/MatMul_1MatMulTanh_1/gradients_1/add_1_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
z
*gradients_1/MatMul_3_grad/tuple/group_depsNoOp!^gradients_1/MatMul_3_grad/MatMul#^gradients_1/MatMul_3_grad/MatMul_1
Ë
2gradients_1/MatMul_3_grad/tuple/control_dependencyIdentity gradients_1/MatMul_3_grad/MatMul+^gradients_1/MatMul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/MatMul_3_grad/MatMul
Ņ
4gradients_1/MatMul_3_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_3_grad/MatMul_1+^gradients_1/MatMul_3_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/MatMul_3_grad/MatMul_1
q
 gradients_1/Tanh_1_grad/TanhGradTanhGradTanh_12gradients_1/MatMul_3_grad/tuple/control_dependency*
T0
H
gradients_1/Add_5_grad/ShapeShapeMatMul_2*
T0*
out_type0
L
gradients_1/Add_5_grad/Shape_1Const*
dtype0*
valueB:d

,gradients_1/Add_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Add_5_grad/Shapegradients_1/Add_5_grad/Shape_1*
T0

gradients_1/Add_5_grad/SumSum gradients_1/Tanh_1_grad/TanhGrad,gradients_1/Add_5_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
z
gradients_1/Add_5_grad/ReshapeReshapegradients_1/Add_5_grad/Sumgradients_1/Add_5_grad/Shape*
T0*
Tshape0

gradients_1/Add_5_grad/Sum_1Sum gradients_1/Tanh_1_grad/TanhGrad.gradients_1/Add_5_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0

 gradients_1/Add_5_grad/Reshape_1Reshapegradients_1/Add_5_grad/Sum_1gradients_1/Add_5_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/Add_5_grad/tuple/group_depsNoOp^gradients_1/Add_5_grad/Reshape!^gradients_1/Add_5_grad/Reshape_1
Á
/gradients_1/Add_5_grad/tuple/control_dependencyIdentitygradients_1/Add_5_grad/Reshape(^gradients_1/Add_5_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Add_5_grad/Reshape
Į
1gradients_1/Add_5_grad/tuple/control_dependency_1Identity gradients_1/Add_5_grad/Reshape_1(^gradients_1/Add_5_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Add_5_grad/Reshape_1

 gradients_1/MatMul_2_grad/MatMulMatMul/gradients_1/Add_5_grad/tuple/control_dependencyVariable_12/read*
T0*
transpose_a( *
transpose_b(

"gradients_1/MatMul_2_grad/MatMul_1MatMul	Reshape_3/gradients_1/Add_5_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
z
*gradients_1/MatMul_2_grad/tuple/group_depsNoOp!^gradients_1/MatMul_2_grad/MatMul#^gradients_1/MatMul_2_grad/MatMul_1
Ë
2gradients_1/MatMul_2_grad/tuple/control_dependencyIdentity gradients_1/MatMul_2_grad/MatMul+^gradients_1/MatMul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/MatMul_2_grad/MatMul
Ņ
4gradients_1/MatMul_2_grad/tuple/control_dependency_1Identity"gradients_1/MatMul_2_grad/MatMul_1+^gradients_1/MatMul_2_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/MatMul_2_grad/MatMul_1
J
 gradients_1/Reshape_3_grad/ShapeShapeRelu_3*
T0*
out_type0

"gradients_1/Reshape_3_grad/ReshapeReshape2gradients_1/MatMul_2_grad/tuple/control_dependency gradients_1/Reshape_3_grad/Shape*
T0*
Tshape0
a
 gradients_1/Relu_3_grad/ReluGradReluGrad"gradients_1/Reshape_3_grad/ReshapeRelu_3*
T0
K
gradients_1/Add_4_grad/ShapeShapedepthwise_3*
T0*
out_type0
L
gradients_1/Add_4_grad/Shape_1Const*
dtype0*
valueB:P

,gradients_1/Add_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Add_4_grad/Shapegradients_1/Add_4_grad/Shape_1*
T0

gradients_1/Add_4_grad/SumSum gradients_1/Relu_3_grad/ReluGrad,gradients_1/Add_4_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
z
gradients_1/Add_4_grad/ReshapeReshapegradients_1/Add_4_grad/Sumgradients_1/Add_4_grad/Shape*
T0*
Tshape0

gradients_1/Add_4_grad/Sum_1Sum gradients_1/Relu_3_grad/ReluGrad.gradients_1/Add_4_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0

 gradients_1/Add_4_grad/Reshape_1Reshapegradients_1/Add_4_grad/Sum_1gradients_1/Add_4_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/Add_4_grad/tuple/group_depsNoOp^gradients_1/Add_4_grad/Reshape!^gradients_1/Add_4_grad/Reshape_1
Á
/gradients_1/Add_4_grad/tuple/control_dependencyIdentitygradients_1/Add_4_grad/Reshape(^gradients_1/Add_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Add_4_grad/Reshape
Į
1gradients_1/Add_4_grad/tuple/control_dependency_1Identity gradients_1/Add_4_grad/Reshape_1(^gradients_1/Add_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Add_4_grad/Reshape_1
O
"gradients_1/depthwise_3_grad/ShapeShape	MaxPool_1*
T0*
out_type0

?gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput"gradients_1/depthwise_3_grad/ShapeVariable_10/read/gradients_1/Add_4_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
a
$gradients_1/depthwise_3_grad/Shape_1Const*
dtype0*%
valueB"      (      

@gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter	MaxPool_1$gradients_1/depthwise_3_grad/Shape_1/gradients_1/Add_4_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
š
-gradients_1/depthwise_3_grad/tuple/group_depsNoOp@^gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropInputA^gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropFilter

5gradients_1/depthwise_3_grad/tuple/control_dependencyIdentity?gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropInput.^gradients_1/depthwise_3_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropInput

7gradients_1/depthwise_3_grad/tuple/control_dependency_1Identity@gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropFilter.^gradients_1/depthwise_3_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/depthwise_3_grad/DepthwiseConv2dNativeBackpropFilter
Û
&gradients_1/MaxPool_1_grad/MaxPoolGradMaxPoolGradRelu_2	MaxPool_15gradients_1/depthwise_3_grad/tuple/control_dependency*
strides
*
paddingVALID*
data_formatNHWC*
ksize
*
T0
e
 gradients_1/Relu_2_grad/ReluGradReluGrad&gradients_1/MaxPool_1_grad/MaxPoolGradRelu_2*
T0
K
gradients_1/Add_3_grad/ShapeShapedepthwise_2*
T0*
out_type0
L
gradients_1/Add_3_grad/Shape_1Const*
dtype0*
valueB:(

,gradients_1/Add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Add_3_grad/Shapegradients_1/Add_3_grad/Shape_1*
T0

gradients_1/Add_3_grad/SumSum gradients_1/Relu_2_grad/ReluGrad,gradients_1/Add_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0
z
gradients_1/Add_3_grad/ReshapeReshapegradients_1/Add_3_grad/Sumgradients_1/Add_3_grad/Shape*
T0*
Tshape0

gradients_1/Add_3_grad/Sum_1Sum gradients_1/Relu_2_grad/ReluGrad.gradients_1/Add_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0

 gradients_1/Add_3_grad/Reshape_1Reshapegradients_1/Add_3_grad/Sum_1gradients_1/Add_3_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/Add_3_grad/tuple/group_depsNoOp^gradients_1/Add_3_grad/Reshape!^gradients_1/Add_3_grad/Reshape_1
Á
/gradients_1/Add_3_grad/tuple/control_dependencyIdentitygradients_1/Add_3_grad/Reshape(^gradients_1/Add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Add_3_grad/Reshape
Į
1gradients_1/Add_3_grad/tuple/control_dependency_1Identity gradients_1/Add_3_grad/Reshape_1(^gradients_1/Add_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Add_3_grad/Reshape_1
O
"gradients_1/depthwise_2_grad/ShapeShape	Reshape_2*
T0*
out_type0

?gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropInput"DepthwiseConv2dNativeBackpropInput"gradients_1/depthwise_2_grad/ShapeVariable_8/read/gradients_1/Add_3_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
a
$gradients_1/depthwise_2_grad/Shape_1Const*
dtype0*%
valueB"            

@gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter	Reshape_2$gradients_1/depthwise_2_grad/Shape_1/gradients_1/Add_3_grad/tuple/control_dependency*
T0*
strides
*
paddingVALID*
data_formatNHWC
š
-gradients_1/depthwise_2_grad/tuple/group_depsNoOp@^gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropInputA^gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropFilter

5gradients_1/depthwise_2_grad/tuple/control_dependencyIdentity?gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropInput.^gradients_1/depthwise_2_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropInput

7gradients_1/depthwise_2_grad/tuple/control_dependency_1Identity@gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropFilter.^gradients_1/depthwise_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients_1/depthwise_2_grad/DepthwiseConv2dNativeBackpropFilter
h
beta1_power_1/initial_valueConst*
dtype0*
_class
loc:@Variable_10*
valueB
 *fff?
y
beta1_power_1
VariableV2*
shape: *
dtype0*
_class
loc:@Variable_10*
shared_name *
	container 

beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
V
beta1_power_1/readIdentitybeta1_power_1*
T0*
_class
loc:@Variable_10
h
beta2_power_1/initial_valueConst*
dtype0*
_class
loc:@Variable_10*
valueB
 *wū?
y
beta2_power_1
VariableV2*
shape: *
dtype0*
_class
loc:@Variable_10*
shared_name *
	container 

beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
V
beta2_power_1/readIdentitybeta2_power_1*
T0*
_class
loc:@Variable_10
}
!Variable_8/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_8*%
valueB*    

Variable_8/Adam
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_8*
shared_name *
	container 
Ĩ
Variable_8/Adam/AssignAssignVariable_8/Adam!Variable_8/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_8*
validate_shape(*
use_locking(
Y
Variable_8/Adam/readIdentityVariable_8/Adam*
T0*
_class
loc:@Variable_8

#Variable_8/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_8*%
valueB*    

Variable_8/Adam_1
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_8*
shared_name *
	container 
Ŧ
Variable_8/Adam_1/AssignAssignVariable_8/Adam_1#Variable_8/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_8*
validate_shape(*
use_locking(
]
Variable_8/Adam_1/readIdentityVariable_8/Adam_1*
T0*
_class
loc:@Variable_8
q
!Variable_9/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_9*
valueB(*    
~
Variable_9/Adam
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_9*
shared_name *
	container 
Ĩ
Variable_9/Adam/AssignAssignVariable_9/Adam!Variable_9/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
Y
Variable_9/Adam/readIdentityVariable_9/Adam*
T0*
_class
loc:@Variable_9
s
#Variable_9/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_9*
valueB(*    

Variable_9/Adam_1
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_9*
shared_name *
	container 
Ŧ
Variable_9/Adam_1/AssignAssignVariable_9/Adam_1#Variable_9/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
]
Variable_9/Adam_1/readIdentityVariable_9/Adam_1*
T0*
_class
loc:@Variable_9

"Variable_10/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_10*%
valueB(*    

Variable_10/Adam
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_10*
shared_name *
	container 
Đ
Variable_10/Adam/AssignAssignVariable_10/Adam"Variable_10/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
\
Variable_10/Adam/readIdentityVariable_10/Adam*
T0*
_class
loc:@Variable_10

$Variable_10/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_10*%
valueB(*    

Variable_10/Adam_1
VariableV2*
shape:(*
dtype0*
_class
loc:@Variable_10*
shared_name *
	container 
Ŋ
Variable_10/Adam_1/AssignAssignVariable_10/Adam_1$Variable_10/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
`
Variable_10/Adam_1/readIdentityVariable_10/Adam_1*
T0*
_class
loc:@Variable_10
s
"Variable_11/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_11*
valueBP*    

Variable_11/Adam
VariableV2*
shape:P*
dtype0*
_class
loc:@Variable_11*
shared_name *
	container 
Đ
Variable_11/Adam/AssignAssignVariable_11/Adam"Variable_11/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_11*
validate_shape(*
use_locking(
\
Variable_11/Adam/readIdentityVariable_11/Adam*
T0*
_class
loc:@Variable_11
u
$Variable_11/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_11*
valueBP*    

Variable_11/Adam_1
VariableV2*
shape:P*
dtype0*
_class
loc:@Variable_11*
shared_name *
	container 
Ŋ
Variable_11/Adam_1/AssignAssignVariable_11/Adam_1$Variable_11/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_11*
validate_shape(*
use_locking(
`
Variable_11/Adam_1/readIdentityVariable_11/Adam_1*
T0*
_class
loc:@Variable_11
x
"Variable_12/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_12*
valueB	ðd*    

Variable_12/Adam
VariableV2*
shape:	ðd*
dtype0*
_class
loc:@Variable_12*
shared_name *
	container 
Đ
Variable_12/Adam/AssignAssignVariable_12/Adam"Variable_12/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_12*
validate_shape(*
use_locking(
\
Variable_12/Adam/readIdentityVariable_12/Adam*
T0*
_class
loc:@Variable_12
z
$Variable_12/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_12*
valueB	ðd*    

Variable_12/Adam_1
VariableV2*
shape:	ðd*
dtype0*
_class
loc:@Variable_12*
shared_name *
	container 
Ŋ
Variable_12/Adam_1/AssignAssignVariable_12/Adam_1$Variable_12/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_12*
validate_shape(*
use_locking(
`
Variable_12/Adam_1/readIdentityVariable_12/Adam_1*
T0*
_class
loc:@Variable_12
s
"Variable_13/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_13*
valueBd*    

Variable_13/Adam
VariableV2*
shape:d*
dtype0*
_class
loc:@Variable_13*
shared_name *
	container 
Đ
Variable_13/Adam/AssignAssignVariable_13/Adam"Variable_13/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
\
Variable_13/Adam/readIdentityVariable_13/Adam*
T0*
_class
loc:@Variable_13
u
$Variable_13/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_13*
valueBd*    

Variable_13/Adam_1
VariableV2*
shape:d*
dtype0*
_class
loc:@Variable_13*
shared_name *
	container 
Ŋ
Variable_13/Adam_1/AssignAssignVariable_13/Adam_1$Variable_13/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
`
Variable_13/Adam_1/readIdentityVariable_13/Adam_1*
T0*
_class
loc:@Variable_13
w
"Variable_14/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_14*
valueBd*    

Variable_14/Adam
VariableV2*
shape
:d*
dtype0*
_class
loc:@Variable_14*
shared_name *
	container 
Đ
Variable_14/Adam/AssignAssignVariable_14/Adam"Variable_14/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_14*
validate_shape(*
use_locking(
\
Variable_14/Adam/readIdentityVariable_14/Adam*
T0*
_class
loc:@Variable_14
y
$Variable_14/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_14*
valueBd*    

Variable_14/Adam_1
VariableV2*
shape
:d*
dtype0*
_class
loc:@Variable_14*
shared_name *
	container 
Ŋ
Variable_14/Adam_1/AssignAssignVariable_14/Adam_1$Variable_14/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_14*
validate_shape(*
use_locking(
`
Variable_14/Adam_1/readIdentityVariable_14/Adam_1*
T0*
_class
loc:@Variable_14
s
"Variable_15/Adam/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_15*
valueB*    

Variable_15/Adam
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_15*
shared_name *
	container 
Đ
Variable_15/Adam/AssignAssignVariable_15/Adam"Variable_15/Adam/Initializer/zeros*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
\
Variable_15/Adam/readIdentityVariable_15/Adam*
T0*
_class
loc:@Variable_15
u
$Variable_15/Adam_1/Initializer/zerosConst*
dtype0*
_class
loc:@Variable_15*
valueB*    

Variable_15/Adam_1
VariableV2*
shape:*
dtype0*
_class
loc:@Variable_15*
shared_name *
	container 
Ŋ
Variable_15/Adam_1/AssignAssignVariable_15/Adam_1$Variable_15/Adam_1/Initializer/zeros*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
`
Variable_15/Adam_1/readIdentityVariable_15/Adam_1*
T0*
_class
loc:@Variable_15
A
Adam_1/learning_rateConst*
dtype0*
valueB
 *·Ņ8
9
Adam_1/beta1Const*
dtype0*
valueB
 *fff?
9
Adam_1/beta2Const*
dtype0*
valueB
 *wū?
;
Adam_1/epsilonConst*
dtype0*
valueB
 *wĖ+2
Ņ
"Adam_1/update_Variable_8/ApplyAdam	ApplyAdam
Variable_8Variable_8/AdamVariable_8/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon7gradients_1/depthwise_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_8*
use_nesterov( *
use_locking( 
Ë
"Adam_1/update_Variable_9/ApplyAdam	ApplyAdam
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon1gradients_1/Add_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_9*
use_nesterov( *
use_locking( 
Ö
#Adam_1/update_Variable_10/ApplyAdam	ApplyAdamVariable_10Variable_10/AdamVariable_10/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon7gradients_1/depthwise_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_10*
use_nesterov( *
use_locking( 
Ð
#Adam_1/update_Variable_11/ApplyAdam	ApplyAdamVariable_11Variable_11/AdamVariable_11/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon1gradients_1/Add_4_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_11*
use_nesterov( *
use_locking( 
Ó
#Adam_1/update_Variable_12/ApplyAdam	ApplyAdamVariable_12Variable_12/AdamVariable_12/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon4gradients_1/MatMul_2_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_12*
use_nesterov( *
use_locking( 
Ð
#Adam_1/update_Variable_13/ApplyAdam	ApplyAdamVariable_13Variable_13/AdamVariable_13/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon1gradients_1/Add_5_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_13*
use_nesterov( *
use_locking( 
Ó
#Adam_1/update_Variable_14/ApplyAdam	ApplyAdamVariable_14Variable_14/AdamVariable_14/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon4gradients_1/MatMul_3_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_14*
use_nesterov( *
use_locking( 
Ð
#Adam_1/update_Variable_15/ApplyAdam	ApplyAdamVariable_15Variable_15/AdamVariable_15/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon1gradients_1/add_1_grad/tuple/control_dependency_1*
T0*
_class
loc:@Variable_15*
use_nesterov( *
use_locking( 


Adam_1/mulMulbeta1_power_1/readAdam_1/beta1#^Adam_1/update_Variable_8/ApplyAdam#^Adam_1/update_Variable_9/ApplyAdam$^Adam_1/update_Variable_10/ApplyAdam$^Adam_1/update_Variable_11/ApplyAdam$^Adam_1/update_Variable_12/ApplyAdam$^Adam_1/update_Variable_13/ApplyAdam$^Adam_1/update_Variable_14/ApplyAdam$^Adam_1/update_Variable_15/ApplyAdam*
T0*
_class
loc:@Variable_10

Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking( 

Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2#^Adam_1/update_Variable_8/ApplyAdam#^Adam_1/update_Variable_9/ApplyAdam$^Adam_1/update_Variable_10/ApplyAdam$^Adam_1/update_Variable_11/ApplyAdam$^Adam_1/update_Variable_12/ApplyAdam$^Adam_1/update_Variable_13/ApplyAdam$^Adam_1/update_Variable_14/ApplyAdam$^Adam_1/update_Variable_15/ApplyAdam*
T0*
_class
loc:@Variable_10

Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking( 
Þ
Adam_1NoOp#^Adam_1/update_Variable_8/ApplyAdam#^Adam_1/update_Variable_9/ApplyAdam$^Adam_1/update_Variable_10/ApplyAdam$^Adam_1/update_Variable_11/ApplyAdam$^Adam_1/update_Variable_12/ApplyAdam$^Adam_1/update_Variable_13/ApplyAdam$^Adam_1/update_Variable_14/ApplyAdam$^Adam_1/update_Variable_15/ApplyAdam^Adam_1/Assign^Adam_1/Assign_1
<
ArgMax_2/dimensionConst*
dtype0*
value	B :
_
ArgMax_2ArgMaxlabels_output_1ArgMax_2/dimension*
T0*
output_type0	*

Tidx0
<
ArgMax_3/dimensionConst*
dtype0*
value	B :
]
ArgMax_3ArgMaxPlaceholder_1ArgMax_3/dimension*
T0*
output_type0	*

Tidx0
-
Equal_1EqualArgMax_2ArgMax_3*
T0	
/
Cast_1CastEqual_1*

SrcT0
*

DstT0
6
Const_11Const*
dtype0*
valueB: 
F
Mean_1MeanCast_1Const_11*
T0*
	keep_dims( *

Tidx0
:
save_1/ConstConst*
dtype0*
valueB Bmodel

save_1/SaveV2/tensor_namesConst*
dtype0*Ų
valueÏBĖ4BVariableBVariable/AdamBVariable/Adam_1B
Variable_1BVariable_1/AdamBVariable_1/Adam_1BVariable_10BVariable_10/AdamBVariable_10/Adam_1BVariable_11BVariable_11/AdamBVariable_11/Adam_1BVariable_12BVariable_12/AdamBVariable_12/Adam_1BVariable_13BVariable_13/AdamBVariable_13/Adam_1BVariable_14BVariable_14/AdamBVariable_14/Adam_1BVariable_15BVariable_15/AdamBVariable_15/Adam_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1B
Variable_3BVariable_3/AdamBVariable_3/Adam_1B
Variable_4BVariable_4/AdamBVariable_4/Adam_1B
Variable_5BVariable_5/AdamBVariable_5/Adam_1B
Variable_6BVariable_6/AdamBVariable_6/Adam_1B
Variable_7BVariable_7/AdamBVariable_7/Adam_1B
Variable_8BVariable_8/AdamBVariable_8/Adam_1B
Variable_9BVariable_9/AdamBVariable_9/Adam_1Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1
ą
save_1/SaveV2/shape_and_slicesConst*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
é
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariableVariable/AdamVariable/Adam_1
Variable_1Variable_1/AdamVariable_1/Adam_1Variable_10Variable_10/AdamVariable_10/Adam_1Variable_11Variable_11/AdamVariable_11/Adam_1Variable_12Variable_12/AdamVariable_12/Adam_1Variable_13Variable_13/AdamVariable_13/Adam_1Variable_14Variable_14/AdamVariable_14/Adam_1Variable_15Variable_15/AdamVariable_15/Adam_1
Variable_2Variable_2/AdamVariable_2/Adam_1
Variable_3Variable_3/AdamVariable_3/Adam_1
Variable_4Variable_4/AdamVariable_4/Adam_1
Variable_5Variable_5/AdamVariable_5/Adam_1
Variable_6Variable_6/AdamVariable_6/Adam_1
Variable_7Variable_7/AdamVariable_7/Adam_1
Variable_8Variable_8/AdamVariable_8/Adam_1
Variable_9Variable_9/AdamVariable_9/Adam_1beta1_powerbeta1_power_1beta2_powerbeta2_power_1*B
dtypes8
624
m
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const
R
save_1/RestoreV2/tensor_namesConst*
dtype0*
valueBBVariable
N
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B 
~
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2

save_1/AssignAssignVariablesave_1/RestoreV2*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
Y
save_1/RestoreV2_1/tensor_namesConst*
dtype0*"
valueBBVariable/Adam
P
#save_1/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2

save_1/Assign_1AssignVariable/Adamsave_1/RestoreV2_1*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
[
save_1/RestoreV2_2/tensor_namesConst*
dtype0*$
valueBBVariable/Adam_1
P
#save_1/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2

save_1/Assign_2AssignVariable/Adam_1save_1/RestoreV2_2*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
V
save_1/RestoreV2_3/tensor_namesConst*
dtype0*
valueBB
Variable_1
P
#save_1/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2

save_1/Assign_3Assign
Variable_1save_1/RestoreV2_3*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
[
save_1/RestoreV2_4/tensor_namesConst*
dtype0*$
valueBBVariable_1/Adam
P
#save_1/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2

save_1/Assign_4AssignVariable_1/Adamsave_1/RestoreV2_4*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
]
save_1/RestoreV2_5/tensor_namesConst*
dtype0*&
valueBBVariable_1/Adam_1
P
#save_1/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2

save_1/Assign_5AssignVariable_1/Adam_1save_1/RestoreV2_5*
T0*
_class
loc:@Variable_1*
validate_shape(*
use_locking(
W
save_1/RestoreV2_6/tensor_namesConst*
dtype0* 
valueBBVariable_10
P
#save_1/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2

save_1/Assign_6AssignVariable_10save_1/RestoreV2_6*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
\
save_1/RestoreV2_7/tensor_namesConst*
dtype0*%
valueBBVariable_10/Adam
P
#save_1/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2

save_1/Assign_7AssignVariable_10/Adamsave_1/RestoreV2_7*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
^
save_1/RestoreV2_8/tensor_namesConst*
dtype0*'
valueBBVariable_10/Adam_1
P
#save_1/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2

save_1/Assign_8AssignVariable_10/Adam_1save_1/RestoreV2_8*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
W
save_1/RestoreV2_9/tensor_namesConst*
dtype0* 
valueBBVariable_11
P
#save_1/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
dtypes
2

save_1/Assign_9AssignVariable_11save_1/RestoreV2_9*
T0*
_class
loc:@Variable_11*
validate_shape(*
use_locking(
]
 save_1/RestoreV2_10/tensor_namesConst*
dtype0*%
valueBBVariable_11/Adam
Q
$save_1/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
dtypes
2

save_1/Assign_10AssignVariable_11/Adamsave_1/RestoreV2_10*
T0*
_class
loc:@Variable_11*
validate_shape(*
use_locking(
_
 save_1/RestoreV2_11/tensor_namesConst*
dtype0*'
valueBBVariable_11/Adam_1
Q
$save_1/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
dtypes
2

save_1/Assign_11AssignVariable_11/Adam_1save_1/RestoreV2_11*
T0*
_class
loc:@Variable_11*
validate_shape(*
use_locking(
X
 save_1/RestoreV2_12/tensor_namesConst*
dtype0* 
valueBBVariable_12
Q
$save_1/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
dtypes
2

save_1/Assign_12AssignVariable_12save_1/RestoreV2_12*
T0*
_class
loc:@Variable_12*
validate_shape(*
use_locking(
]
 save_1/RestoreV2_13/tensor_namesConst*
dtype0*%
valueBBVariable_12/Adam
Q
$save_1/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
dtypes
2

save_1/Assign_13AssignVariable_12/Adamsave_1/RestoreV2_13*
T0*
_class
loc:@Variable_12*
validate_shape(*
use_locking(
_
 save_1/RestoreV2_14/tensor_namesConst*
dtype0*'
valueBBVariable_12/Adam_1
Q
$save_1/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2

save_1/Assign_14AssignVariable_12/Adam_1save_1/RestoreV2_14*
T0*
_class
loc:@Variable_12*
validate_shape(*
use_locking(
X
 save_1/RestoreV2_15/tensor_namesConst*
dtype0* 
valueBBVariable_13
Q
$save_1/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2

save_1/Assign_15AssignVariable_13save_1/RestoreV2_15*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
]
 save_1/RestoreV2_16/tensor_namesConst*
dtype0*%
valueBBVariable_13/Adam
Q
$save_1/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
dtypes
2

save_1/Assign_16AssignVariable_13/Adamsave_1/RestoreV2_16*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
_
 save_1/RestoreV2_17/tensor_namesConst*
dtype0*'
valueBBVariable_13/Adam_1
Q
$save_1/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
dtypes
2

save_1/Assign_17AssignVariable_13/Adam_1save_1/RestoreV2_17*
T0*
_class
loc:@Variable_13*
validate_shape(*
use_locking(
X
 save_1/RestoreV2_18/tensor_namesConst*
dtype0* 
valueBBVariable_14
Q
$save_1/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
dtypes
2

save_1/Assign_18AssignVariable_14save_1/RestoreV2_18*
T0*
_class
loc:@Variable_14*
validate_shape(*
use_locking(
]
 save_1/RestoreV2_19/tensor_namesConst*
dtype0*%
valueBBVariable_14/Adam
Q
$save_1/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
dtypes
2

save_1/Assign_19AssignVariable_14/Adamsave_1/RestoreV2_19*
T0*
_class
loc:@Variable_14*
validate_shape(*
use_locking(
_
 save_1/RestoreV2_20/tensor_namesConst*
dtype0*'
valueBBVariable_14/Adam_1
Q
$save_1/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_20	RestoreV2save_1/Const save_1/RestoreV2_20/tensor_names$save_1/RestoreV2_20/shape_and_slices*
dtypes
2

save_1/Assign_20AssignVariable_14/Adam_1save_1/RestoreV2_20*
T0*
_class
loc:@Variable_14*
validate_shape(*
use_locking(
X
 save_1/RestoreV2_21/tensor_namesConst*
dtype0* 
valueBBVariable_15
Q
$save_1/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_21	RestoreV2save_1/Const save_1/RestoreV2_21/tensor_names$save_1/RestoreV2_21/shape_and_slices*
dtypes
2

save_1/Assign_21AssignVariable_15save_1/RestoreV2_21*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
]
 save_1/RestoreV2_22/tensor_namesConst*
dtype0*%
valueBBVariable_15/Adam
Q
$save_1/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_22	RestoreV2save_1/Const save_1/RestoreV2_22/tensor_names$save_1/RestoreV2_22/shape_and_slices*
dtypes
2

save_1/Assign_22AssignVariable_15/Adamsave_1/RestoreV2_22*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
_
 save_1/RestoreV2_23/tensor_namesConst*
dtype0*'
valueBBVariable_15/Adam_1
Q
$save_1/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_23	RestoreV2save_1/Const save_1/RestoreV2_23/tensor_names$save_1/RestoreV2_23/shape_and_slices*
dtypes
2

save_1/Assign_23AssignVariable_15/Adam_1save_1/RestoreV2_23*
T0*
_class
loc:@Variable_15*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_24/tensor_namesConst*
dtype0*
valueBB
Variable_2
Q
$save_1/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_24	RestoreV2save_1/Const save_1/RestoreV2_24/tensor_names$save_1/RestoreV2_24/shape_and_slices*
dtypes
2

save_1/Assign_24Assign
Variable_2save_1/RestoreV2_24*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_25/tensor_namesConst*
dtype0*$
valueBBVariable_2/Adam
Q
$save_1/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_25	RestoreV2save_1/Const save_1/RestoreV2_25/tensor_names$save_1/RestoreV2_25/shape_and_slices*
dtypes
2

save_1/Assign_25AssignVariable_2/Adamsave_1/RestoreV2_25*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_26/tensor_namesConst*
dtype0*&
valueBBVariable_2/Adam_1
Q
$save_1/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_26	RestoreV2save_1/Const save_1/RestoreV2_26/tensor_names$save_1/RestoreV2_26/shape_and_slices*
dtypes
2

save_1/Assign_26AssignVariable_2/Adam_1save_1/RestoreV2_26*
T0*
_class
loc:@Variable_2*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_27/tensor_namesConst*
dtype0*
valueBB
Variable_3
Q
$save_1/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_27	RestoreV2save_1/Const save_1/RestoreV2_27/tensor_names$save_1/RestoreV2_27/shape_and_slices*
dtypes
2

save_1/Assign_27Assign
Variable_3save_1/RestoreV2_27*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_28/tensor_namesConst*
dtype0*$
valueBBVariable_3/Adam
Q
$save_1/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_28	RestoreV2save_1/Const save_1/RestoreV2_28/tensor_names$save_1/RestoreV2_28/shape_and_slices*
dtypes
2

save_1/Assign_28AssignVariable_3/Adamsave_1/RestoreV2_28*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_29/tensor_namesConst*
dtype0*&
valueBBVariable_3/Adam_1
Q
$save_1/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_29	RestoreV2save_1/Const save_1/RestoreV2_29/tensor_names$save_1/RestoreV2_29/shape_and_slices*
dtypes
2

save_1/Assign_29AssignVariable_3/Adam_1save_1/RestoreV2_29*
T0*
_class
loc:@Variable_3*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_30/tensor_namesConst*
dtype0*
valueBB
Variable_4
Q
$save_1/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_30	RestoreV2save_1/Const save_1/RestoreV2_30/tensor_names$save_1/RestoreV2_30/shape_and_slices*
dtypes
2

save_1/Assign_30Assign
Variable_4save_1/RestoreV2_30*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_31/tensor_namesConst*
dtype0*$
valueBBVariable_4/Adam
Q
$save_1/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_31	RestoreV2save_1/Const save_1/RestoreV2_31/tensor_names$save_1/RestoreV2_31/shape_and_slices*
dtypes
2

save_1/Assign_31AssignVariable_4/Adamsave_1/RestoreV2_31*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_32/tensor_namesConst*
dtype0*&
valueBBVariable_4/Adam_1
Q
$save_1/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_32	RestoreV2save_1/Const save_1/RestoreV2_32/tensor_names$save_1/RestoreV2_32/shape_and_slices*
dtypes
2

save_1/Assign_32AssignVariable_4/Adam_1save_1/RestoreV2_32*
T0*
_class
loc:@Variable_4*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_33/tensor_namesConst*
dtype0*
valueBB
Variable_5
Q
$save_1/RestoreV2_33/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_33	RestoreV2save_1/Const save_1/RestoreV2_33/tensor_names$save_1/RestoreV2_33/shape_and_slices*
dtypes
2

save_1/Assign_33Assign
Variable_5save_1/RestoreV2_33*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_34/tensor_namesConst*
dtype0*$
valueBBVariable_5/Adam
Q
$save_1/RestoreV2_34/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_34	RestoreV2save_1/Const save_1/RestoreV2_34/tensor_names$save_1/RestoreV2_34/shape_and_slices*
dtypes
2

save_1/Assign_34AssignVariable_5/Adamsave_1/RestoreV2_34*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_35/tensor_namesConst*
dtype0*&
valueBBVariable_5/Adam_1
Q
$save_1/RestoreV2_35/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_35	RestoreV2save_1/Const save_1/RestoreV2_35/tensor_names$save_1/RestoreV2_35/shape_and_slices*
dtypes
2

save_1/Assign_35AssignVariable_5/Adam_1save_1/RestoreV2_35*
T0*
_class
loc:@Variable_5*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_36/tensor_namesConst*
dtype0*
valueBB
Variable_6
Q
$save_1/RestoreV2_36/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_36	RestoreV2save_1/Const save_1/RestoreV2_36/tensor_names$save_1/RestoreV2_36/shape_and_slices*
dtypes
2

save_1/Assign_36Assign
Variable_6save_1/RestoreV2_36*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_37/tensor_namesConst*
dtype0*$
valueBBVariable_6/Adam
Q
$save_1/RestoreV2_37/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_37	RestoreV2save_1/Const save_1/RestoreV2_37/tensor_names$save_1/RestoreV2_37/shape_and_slices*
dtypes
2

save_1/Assign_37AssignVariable_6/Adamsave_1/RestoreV2_37*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_38/tensor_namesConst*
dtype0*&
valueBBVariable_6/Adam_1
Q
$save_1/RestoreV2_38/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_38	RestoreV2save_1/Const save_1/RestoreV2_38/tensor_names$save_1/RestoreV2_38/shape_and_slices*
dtypes
2

save_1/Assign_38AssignVariable_6/Adam_1save_1/RestoreV2_38*
T0*
_class
loc:@Variable_6*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_39/tensor_namesConst*
dtype0*
valueBB
Variable_7
Q
$save_1/RestoreV2_39/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_39	RestoreV2save_1/Const save_1/RestoreV2_39/tensor_names$save_1/RestoreV2_39/shape_and_slices*
dtypes
2

save_1/Assign_39Assign
Variable_7save_1/RestoreV2_39*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_40/tensor_namesConst*
dtype0*$
valueBBVariable_7/Adam
Q
$save_1/RestoreV2_40/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_40	RestoreV2save_1/Const save_1/RestoreV2_40/tensor_names$save_1/RestoreV2_40/shape_and_slices*
dtypes
2

save_1/Assign_40AssignVariable_7/Adamsave_1/RestoreV2_40*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_41/tensor_namesConst*
dtype0*&
valueBBVariable_7/Adam_1
Q
$save_1/RestoreV2_41/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_41	RestoreV2save_1/Const save_1/RestoreV2_41/tensor_names$save_1/RestoreV2_41/shape_and_slices*
dtypes
2

save_1/Assign_41AssignVariable_7/Adam_1save_1/RestoreV2_41*
T0*
_class
loc:@Variable_7*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_42/tensor_namesConst*
dtype0*
valueBB
Variable_8
Q
$save_1/RestoreV2_42/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_42	RestoreV2save_1/Const save_1/RestoreV2_42/tensor_names$save_1/RestoreV2_42/shape_and_slices*
dtypes
2

save_1/Assign_42Assign
Variable_8save_1/RestoreV2_42*
T0*
_class
loc:@Variable_8*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_43/tensor_namesConst*
dtype0*$
valueBBVariable_8/Adam
Q
$save_1/RestoreV2_43/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_43	RestoreV2save_1/Const save_1/RestoreV2_43/tensor_names$save_1/RestoreV2_43/shape_and_slices*
dtypes
2

save_1/Assign_43AssignVariable_8/Adamsave_1/RestoreV2_43*
T0*
_class
loc:@Variable_8*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_44/tensor_namesConst*
dtype0*&
valueBBVariable_8/Adam_1
Q
$save_1/RestoreV2_44/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_44	RestoreV2save_1/Const save_1/RestoreV2_44/tensor_names$save_1/RestoreV2_44/shape_and_slices*
dtypes
2

save_1/Assign_44AssignVariable_8/Adam_1save_1/RestoreV2_44*
T0*
_class
loc:@Variable_8*
validate_shape(*
use_locking(
W
 save_1/RestoreV2_45/tensor_namesConst*
dtype0*
valueBB
Variable_9
Q
$save_1/RestoreV2_45/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_45	RestoreV2save_1/Const save_1/RestoreV2_45/tensor_names$save_1/RestoreV2_45/shape_and_slices*
dtypes
2

save_1/Assign_45Assign
Variable_9save_1/RestoreV2_45*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
\
 save_1/RestoreV2_46/tensor_namesConst*
dtype0*$
valueBBVariable_9/Adam
Q
$save_1/RestoreV2_46/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_46	RestoreV2save_1/Const save_1/RestoreV2_46/tensor_names$save_1/RestoreV2_46/shape_and_slices*
dtypes
2

save_1/Assign_46AssignVariable_9/Adamsave_1/RestoreV2_46*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
^
 save_1/RestoreV2_47/tensor_namesConst*
dtype0*&
valueBBVariable_9/Adam_1
Q
$save_1/RestoreV2_47/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_47	RestoreV2save_1/Const save_1/RestoreV2_47/tensor_names$save_1/RestoreV2_47/shape_and_slices*
dtypes
2

save_1/Assign_47AssignVariable_9/Adam_1save_1/RestoreV2_47*
T0*
_class
loc:@Variable_9*
validate_shape(*
use_locking(
X
 save_1/RestoreV2_48/tensor_namesConst*
dtype0* 
valueBBbeta1_power
Q
$save_1/RestoreV2_48/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_48	RestoreV2save_1/Const save_1/RestoreV2_48/tensor_names$save_1/RestoreV2_48/shape_and_slices*
dtypes
2

save_1/Assign_48Assignbeta1_powersave_1/RestoreV2_48*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
Z
 save_1/RestoreV2_49/tensor_namesConst*
dtype0*"
valueBBbeta1_power_1
Q
$save_1/RestoreV2_49/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_49	RestoreV2save_1/Const save_1/RestoreV2_49/tensor_names$save_1/RestoreV2_49/shape_and_slices*
dtypes
2

save_1/Assign_49Assignbeta1_power_1save_1/RestoreV2_49*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
X
 save_1/RestoreV2_50/tensor_namesConst*
dtype0* 
valueBBbeta2_power
Q
$save_1/RestoreV2_50/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_50	RestoreV2save_1/Const save_1/RestoreV2_50/tensor_names$save_1/RestoreV2_50/shape_and_slices*
dtypes
2

save_1/Assign_50Assignbeta2_powersave_1/RestoreV2_50*
T0*
_class
loc:@Variable*
validate_shape(*
use_locking(
Z
 save_1/RestoreV2_51/tensor_namesConst*
dtype0*"
valueBBbeta2_power_1
Q
$save_1/RestoreV2_51/shape_and_slicesConst*
dtype0*
valueB
B 

save_1/RestoreV2_51	RestoreV2save_1/Const save_1/RestoreV2_51/tensor_names$save_1/RestoreV2_51/shape_and_slices*
dtypes
2

save_1/Assign_51Assignbeta2_power_1save_1/RestoreV2_51*
T0*
_class
loc:@Variable_10*
validate_shape(*
use_locking(
ę
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_50^save_1/Assign_51
ō	
init_1NoOp^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign^Variable_5/Assign^Variable_6/Assign^Variable_7/Assign^beta1_power/Assign^beta2_power/Assign^Variable/Adam/Assign^Variable/Adam_1/Assign^Variable_1/Adam/Assign^Variable_1/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign^Variable_3/Adam/Assign^Variable_3/Adam_1/Assign^Variable_4/Adam/Assign^Variable_4/Adam_1/Assign^Variable_5/Adam/Assign^Variable_5/Adam_1/Assign^Variable_6/Adam/Assign^Variable_6/Adam_1/Assign^Variable_7/Adam/Assign^Variable_7/Adam_1/Assign^Variable_8/Assign^Variable_9/Assign^Variable_10/Assign^Variable_11/Assign^Variable_12/Assign^Variable_13/Assign^Variable_14/Assign^Variable_15/Assign^beta1_power_1/Assign^beta2_power_1/Assign^Variable_8/Adam/Assign^Variable_8/Adam_1/Assign^Variable_9/Adam/Assign^Variable_9/Adam_1/Assign^Variable_10/Adam/Assign^Variable_10/Adam_1/Assign^Variable_11/Adam/Assign^Variable_11/Adam_1/Assign^Variable_12/Adam/Assign^Variable_12/Adam_1/Assign^Variable_13/Adam/Assign^Variable_13/Adam_1/Assign^Variable_14/Adam/Assign^Variable_14/Adam_1/Assign^Variable_15/Adam/Assign^Variable_15/Adam_1/Assign"