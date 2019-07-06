import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

start_sentence = "Automatically generating coherent and semantically meaningful text has many applications"

def model_output(start_sentence):
    # start_sentence = "Most state-of-the-art text detection methods are specific to horizontal Latin text and are not fast enough for real-time applications. We introduce Segment Linking (SegLink), an oriented text detection method."
    #
    single_text = gpt2.generate(sess,
        length = 200,
        temperature = 0.7,
        prefix = start_sentence,
        #   nsamples = 4,
        #   batch_size = 5,
        return_as_list=True
        )[0]

    return single_text

# single_text = gpt2.generate(sess, return_as_list=True)[0]
print(model_output(start_sentence))