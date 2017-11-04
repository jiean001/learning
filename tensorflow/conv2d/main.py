import tensorflow as tf

input = tf.Variable(tf.random_normal([1, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 32]))
strides = [1, 1, 1, 1]

op = tf.nn.conv2d(input, filter, strides, padding="SAME")
initial = tf.initialize_all_variables()

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

op_pool = max_pool_2x2(op)

filter2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))

op2 = tf.nn.conv2d(op_pool, filter2, strides, padding="SAME")
op2_pool = max_pool_2x2(op2)



with tf.Session() as sess:
    # sess.run(initial)
    print("input: %s" %input)    
    #print(sess.run(input))
    print("filter:%s" %filter)
    #print(sess.run(filter))
    print("op: %s" %op)
    print("op_pool: %s" %op_pool)
    #print(sess.run(op))
    print("op2: %s" %op2)
    print("op2_pool: %s" %op2_pool)

