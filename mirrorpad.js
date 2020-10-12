

function mirrorPadSymmetric(node) {
	// Author: Ablaikhan Akhazhanov | gkcalat@ucla.edu
	// 
	// Implements SYMMETRIC MirrorPad layer of tfjs
	// 
	// So far, input node does not have "attr" and only contains "inputs" of length=2:
	// 1) input tensor, 2) "paddings" tensor of size [N, 2], where N is the number of 
	// dimensions of the input tensor
	//
	// To convert a saved TF model, you need to include the "--skip_op_check" flag
	//
	// Tested on Tensorflow 2.3.1 and tfjs-v2.6.0 with padding of 4D tensors

	let x = node.inputs[0];
	let idx = node.inputs[1].dataSync();
		
	const input_shape = x.shape;
	let temp_ind, temp_slice, temp_axis;

	for (let i=0; i<idx.length; i+=2){
		temp_axis = Math.floor(i/2);
		if (idx[i+1] > 0){
			temp_ind = tf.range(input_shape[temp_axis]-1, input_shape[temp_axis]-1-idx[i+1], -1, 'int32');
			temp_slice = x.gather(temp_ind, temp_axis);
			x = tf.concat([x, temp_slice], temp_axis);
		}
		if (idx[i] > 0){
			temp_ind = tf.range(0, idx[i], 1, 'int32');
			temp_slice = x.gather(temp_ind, temp_axis);
			x = tf.concat([temp_slice, x], temp_axis);
		}
	}

	for (let i=0; i<idx.length; i+=2){
		temp_axis = Math.floor(i/2);
		if ((input_shape[temp_axis] + idx[i] + idx[i+1]) != x.shape[temp_axis]){
			console.error('Inconsistent output shape in axis ' + i);
		}
	}

	return x;
}


function mirrorPadReflect(node) {
	// Author: Ablaikhan Akhazhanov | gkcalat@ucla.edu
	// 
	// Implements REFLECT MirrorPad layer of tfjs
	// 
	// So far, input node does not have "attr" and only contains "inputs" of length=2:
	// 1) input tensor, 2) "paddings" tensor of size [N, 2], where N is the number of 
	// dimensions of the input tensor
	//
	// To convert a saved TF model, you need to include the "--skip_op_check" flag
	//
	// Tested on Tensorflow 2.3.1 and tfjs-v2.6.0 with padding of 4D tensors

	let x = node.inputs[0];
	let idx = node.inputs[1].dataSync();
		
	const input_shape = x.shape;
	let temp_ind, temp_slice, temp_axis;

	for (let i=0; i<idx.length; i+=2){
		temp_axis = Math.floor(i/2);
		if (idx[i+1] > 0){
			temp_ind = tf.range(input_shape[temp_axis]-2, input_shape[temp_axis]-2-idx[i+1], -1, 'int32');
			temp_slice = x.gather(temp_ind, temp_axis);
			x = tf.concat([x, temp_slice], temp_axis);
		}
		if (idx[i] > 0){
			temp_ind = tf.range(1, idx[i]+1, 1, 'int32');
			temp_slice = x.gather(temp_ind, temp_axis);
			x = tf.concat([temp_slice, x], temp_axis);
		}
	}

	for (let i=0; i<idx.length; i+=2){
		temp_axis = Math.floor(i/2);
		if ((input_shape[temp_axis] + idx[i] + idx[i+1]) != x.shape[temp_axis]){
			console.error('Inconsistent output shape in axis ' + i);
		}
	}

	return x;
}


tf.registerOp('MirrorPad', mirrorPadSymmetric);
