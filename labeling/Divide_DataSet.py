#!/usr/bin/env python
# coding: utf-8

# In[6]:


import splitfolders
import shutil
def divide(_input, _output):

    shutil.rmtree(r".\labeling\divided_output")

    splitfolders.ratio(_input, output=_output + "/", seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values
    print("Dividing images into folders.")
    return


# In[ ]:




