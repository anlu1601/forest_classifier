#!/usr/bin/env python
# coding: utf-8

# In[1]:


import splitfolders
def divide(_input, _output):
#     splitfolders.ratio("dataset", output="after_divide_dataset/", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values
    splitfolders.ratio(_input, output=_output + "/", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values
    print("Dividing images into folders.")
    return


# In[ ]:




