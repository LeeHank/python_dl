??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
?
&my_sequential_model/flexible_dense_2/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&my_sequential_model/flexible_dense_2/w
?
:my_sequential_model/flexible_dense_2/w/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_2/w*
_output_shapes

:*
dtype0
?
&my_sequential_model/flexible_dense_2/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&my_sequential_model/flexible_dense_2/b
?
:my_sequential_model/flexible_dense_2/b/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_2/b*
_output_shapes
:*
dtype0
?
&my_sequential_model/flexible_dense_3/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&my_sequential_model/flexible_dense_3/w
?
:my_sequential_model/flexible_dense_3/w/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_3/w*
_output_shapes

:*
dtype0
?
&my_sequential_model/flexible_dense_3/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&my_sequential_model/flexible_dense_3/b
?
:my_sequential_model/flexible_dense_3/b/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_3/b*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
|
dense_1
dense_2
regularization_losses
	variables
trainable_variables
	keras_api

signatures
`
w
	b

regularization_losses
	variables
trainable_variables
	keras_api
`
w
b
regularization_losses
	variables
trainable_variables
	keras_api
 

0
	1
2
3

0
	1
2
3
?

layers
regularization_losses
metrics
layer_regularization_losses
	variables
layer_metrics
non_trainable_variables
trainable_variables
 
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_2/w$dense_1/w/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_2/b$dense_1/b/.ATTRIBUTES/VARIABLE_VALUE
 

0
	1

0
	1
?

layers

regularization_losses
metrics
layer_regularization_losses
	variables
layer_metrics
non_trainable_variables
trainable_variables
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_3/w$dense_2/w/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_3/b$dense_2/b/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?

layers
regularization_losses
metrics
 layer_regularization_losses
	variables
!layer_metrics
"non_trainable_variables
trainable_variables

0
1
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
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1&my_sequential_model/flexible_dense_2/w&my_sequential_model/flexible_dense_2/b&my_sequential_model/flexible_dense_3/w&my_sequential_model/flexible_dense_3/b*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference_signature_wrapper_768
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:my_sequential_model/flexible_dense_2/w/Read/ReadVariableOp:my_sequential_model/flexible_dense_2/b/Read/ReadVariableOp:my_sequential_model/flexible_dense_3/w/Read/ReadVariableOp:my_sequential_model/flexible_dense_3/b/Read/ReadVariableOpConst*
Tin

2*
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
GPU 2J 8? *%
f R
__inference__traced_save_841
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&my_sequential_model/flexible_dense_2/w&my_sequential_model/flexible_dense_2/b&my_sequential_model/flexible_dense_3/w&my_sequential_model/flexible_dense_3/b*
Tin	
2*
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
__inference__traced_restore_863؍
?
?
__inference__traced_restore_863
file_prefix;
7assignvariableop_my_sequential_model_flexible_dense_2_w=
9assignvariableop_1_my_sequential_model_flexible_dense_2_b=
9assignvariableop_2_my_sequential_model_flexible_dense_3_w=
9assignvariableop_3_my_sequential_model_flexible_dense_3_b

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B$dense_1/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_1/b/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/b/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp7assignvariableop_my_sequential_model_flexible_dense_2_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp9assignvariableop_1_my_sequential_model_flexible_dense_2_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp9assignvariableop_2_my_sequential_model_flexible_dense_3_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp9assignvariableop_3_my_sequential_model_flexible_dense_3_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference__wrapped_model_682
input_1G
Cmy_sequential_model_flexible_dense_2_matmul_readvariableop_resourceD
@my_sequential_model_flexible_dense_2_add_readvariableop_resourceG
Cmy_sequential_model_flexible_dense_3_matmul_readvariableop_resourceD
@my_sequential_model_flexible_dense_3_add_readvariableop_resource
identity??
:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOpReadVariableOpCmy_sequential_model_flexible_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOp?
+my_sequential_model/flexible_dense_2/MatMulMatMulinput_1Bmy_sequential_model/flexible_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+my_sequential_model/flexible_dense_2/MatMul?
7my_sequential_model/flexible_dense_2/add/ReadVariableOpReadVariableOp@my_sequential_model_flexible_dense_2_add_readvariableop_resource*
_output_shapes
:*
dtype029
7my_sequential_model/flexible_dense_2/add/ReadVariableOp?
(my_sequential_model/flexible_dense_2/addAddV25my_sequential_model/flexible_dense_2/MatMul:product:0?my_sequential_model/flexible_dense_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(my_sequential_model/flexible_dense_2/add?
:my_sequential_model/flexible_dense_3/MatMul/ReadVariableOpReadVariableOpCmy_sequential_model_flexible_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:my_sequential_model/flexible_dense_3/MatMul/ReadVariableOp?
+my_sequential_model/flexible_dense_3/MatMulMatMul,my_sequential_model/flexible_dense_2/add:z:0Bmy_sequential_model/flexible_dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+my_sequential_model/flexible_dense_3/MatMul?
7my_sequential_model/flexible_dense_3/add/ReadVariableOpReadVariableOp@my_sequential_model_flexible_dense_3_add_readvariableop_resource*
_output_shapes
:*
dtype029
7my_sequential_model/flexible_dense_3/add/ReadVariableOp?
(my_sequential_model/flexible_dense_3/addAddV25my_sequential_model/flexible_dense_3/MatMul:product:0?my_sequential_model/flexible_dense_3/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(my_sequential_model/flexible_dense_3/add?
IdentityIdentity,my_sequential_model/flexible_dense_3/add:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_778

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_696

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_739
input_1
flexible_dense_2_707
flexible_dense_2_709
flexible_dense_3_733
flexible_dense_3_735
identity??(flexible_dense_2/StatefulPartitionedCall?(flexible_dense_3/StatefulPartitionedCall?
(flexible_dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_1flexible_dense_2_707flexible_dense_2_709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_6962*
(flexible_dense_2/StatefulPartitionedCall?
(flexible_dense_3/StatefulPartitionedCallStatefulPartitionedCall1flexible_dense_2/StatefulPartitionedCall:output:0flexible_dense_3_733flexible_dense_3_735*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flexible_dense_3_layer_call_and_return_conditional_losses_7222*
(flexible_dense_3/StatefulPartitionedCall?
IdentityIdentity1flexible_dense_3/StatefulPartitionedCall:output:0)^flexible_dense_2/StatefulPartitionedCall)^flexible_dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2T
(flexible_dense_2/StatefulPartitionedCall(flexible_dense_2/StatefulPartitionedCall2T
(flexible_dense_3/StatefulPartitionedCall(flexible_dense_3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
!__inference_signature_wrapper_768
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_6822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
.__inference_flexible_dense_2_layer_call_fn_787

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_6962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_flexible_dense_3_layer_call_and_return_conditional_losses_722

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_flexible_dense_3_layer_call_fn_806

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_flexible_dense_3_layer_call_and_return_conditional_losses_7222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_841
file_prefixE
Asavev2_my_sequential_model_flexible_dense_2_w_read_readvariableopE
Asavev2_my_sequential_model_flexible_dense_2_b_read_readvariableopE
Asavev2_my_sequential_model_flexible_dense_3_w_read_readvariableopE
Asavev2_my_sequential_model_flexible_dense_3_b_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_22f0458a1ad44cc2b6f1d0ef405e95ed/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B$dense_1/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_1/b/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/b/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_my_sequential_model_flexible_dense_2_w_read_readvariableopAsavev2_my_sequential_model_flexible_dense_2_b_read_readvariableopAsavev2_my_sequential_model_flexible_dense_3_w_read_readvariableopAsavev2_my_sequential_model_flexible_dense_3_b_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
1__inference_my_sequential_model_layer_call_fn_753
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_7392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
I__inference_flexible_dense_3_layer_call_and_return_conditional_losses_797

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?4
?
dense_1
dense_2
regularization_losses
	variables
trainable_variables
	keras_api

signatures
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature"?
_tf_keras_model?{"class_name": "MySequentialModel", "name": "my_sequential_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MySequentialModel"}}
?
w
	b

regularization_losses
	variables
trainable_variables
	keras_api
&__call__
*'&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "FlexibleDense", "name": "flexible_dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3]}}
?
w
b
regularization_losses
	variables
trainable_variables
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "FlexibleDense", "name": "flexible_dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3]}}
 "
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
?

layers
regularization_losses
metrics
layer_regularization_losses
	variables
layer_metrics
non_trainable_variables
trainable_variables
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
8:62&my_sequential_model/flexible_dense_2/w
4:22&my_sequential_model/flexible_dense_2/b
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?

layers

regularization_losses
metrics
layer_regularization_losses
	variables
layer_metrics
non_trainable_variables
trainable_variables
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
8:62&my_sequential_model/flexible_dense_3/w
4:22&my_sequential_model/flexible_dense_3/b
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

layers
regularization_losses
metrics
 layer_regularization_losses
	variables
!layer_metrics
"non_trainable_variables
trainable_variables
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
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
?2?
1__inference_my_sequential_model_layer_call_fn_753?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_739?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
__inference__wrapped_model_682?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
.__inference_flexible_dense_2_layer_call_fn_787?
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
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_778?
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
.__inference_flexible_dense_3_layer_call_fn_806?
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
I__inference_flexible_dense_3_layer_call_and_return_conditional_losses_797?
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
0B.
!__inference_signature_wrapper_768input_1?
__inference__wrapped_model_682m	0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_778\	/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
.__inference_flexible_dense_2_layer_call_fn_787O	/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_flexible_dense_3_layer_call_and_return_conditional_losses_797\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
.__inference_flexible_dense_3_layer_call_fn_806O/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_739_	0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? ?
1__inference_my_sequential_model_layer_call_fn_753R	0?-
&?#
!?
input_1?????????
? "???????????
!__inference_signature_wrapper_768x	;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????