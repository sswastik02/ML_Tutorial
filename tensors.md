# Tensors

#### watch this [video](https://www.youtube.com/watch?v=f5liqUk0ZTw)
<img src="https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg" alt = "https://www.kdnuggets.com/wp-content/uploads/scalar-vector-matrix-tensor.jpg">

## In Tensorflow python

### Example

```py
import tensorflow as tf

rank0_tensor = tf.Variable("Test",tf.string)
rank1_tensor = tf.Variable(["Test","Test1"].tf.string)
rank2_tensor = tf.Variable([["Test","Test1","Test2"],["TestArgs","TestArgs1","TestArgs2"]],tf.string)

```

## Shape of Tensors

It is basically how many elements in a particular dimension

```py
rank2_tensor.shape #=========> TensorShape([2,3])  2 dimensions with each dimension having 3 elements
```

### Changing shape of tensors

```py
tensor1 = tf.ones([1,2,3]) # fill the tensors with 1. (float type) with the shape [1,2,3]
tensor2 = tf.reshape(tensor1,[2,3,1]) # valid as long as product of shape is the same across all shapes (in this case 6)
tensor3 = tf.reshape(tensor1,[3,-1]) # -1 basically means figure the dimension out ====> in this case it has to be 2 as 3*2 = 6 [3,2]
```

```

tensor1 = 
                    [
                        [
                            [1. 1. 1.] , 
                            [1. 1. 1.]
                        ]
                    ]
tensor2 = 
                    [
                        [
                            [1.],
                            [1.],
                            [1.]
                        ],
                        [
                            [1.],
                            [1.],
                            [1.]
                        ]
                    ]
tensor3 = 
                    [
                        [1. 1.],
                        [1. 1.],
                        [1. 1.]
                    ]

```

## Types of tensors

* Variable => mutable tensor whose value can be changed at anytime
* Constant => immutable and cant be changed
* Placeholder => immutable
* SparseTensor => immutable
  
