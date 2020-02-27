### lifq python package

Installation
============
  
  #Step 0 	

  Donwload the project. This can be done directly by using the green download button. (If you do it this way don't forget to unpack the compressed folder)  
  It can also be done through a command prompt by writing the command line : **git clone github.com/aloaah/flifq** 
  After this step you should have the project folder on your machine.

  #Step 1 

  Go to the folder where the repository is cloned
  run : pip install .


Requirements
============
lifq requires numpy, brian2 and scikit_image_.

.. _numpy = https://github.com/numpy/numpy
.. _brian2 = https://github.com/brian-team/brian2
.. _ scikit_image = https://github.com/scikit-image/scikit-image


Usage
============

To basically apply the Lif quantizer to a signal in 2D : 

.. code-block::python

from lifq import lifq_2d
import matplotlib.pyplot as plt

img = plt.readimg("path/to/img")
lif = lifq_2d()
lif.fit(img)
reconstr_img = lif.getDecodedSignal()

plt.title('Image after LIFQ')
plt.plot(reconstr_img, 'cmap = gray')
plt.show()


Links
======

.. _LIFq paper : https://hal.archives-ouvertes.fr/hal-01650750