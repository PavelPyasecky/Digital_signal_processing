#!/usr/bin/env python
# coding: utf-8

# # Лабораторная работы №4
# ## Обработка биомедицинского изображения на языке Python.
# ***

# ### 1. Подключение библиотек.

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Получение фантома томографии головы.

# In[15]:


from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', multichannel=False)


# ### 3. Алгоритмы обработки фантома.

# ### The forward transfor

# In[16]:


from skimage.transform import radon, rescale

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()
plt.show()


# ### Reconstruction with the Filtered Back Projection (FBP) (метод фильтрованных обратных проекций)

# In[17]:


from skimage.transform import iradon

reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
error = reconstruction_fbp - image
print(f"FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}")

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()


# ### Reconstruction with the Simultaneous Algebraic Reconstruction Technique (реконструкция с одновременными итерациями)

# In[18]:


from skimage.transform import iradon_sart

reconstruction_sart = iradon_sart(sinogram, theta=theta)
error = reconstruction_sart - image
print("SART (1 iteration) rms reconstruction error: "
      f"{np.sqrt(np.mean(error**2)):.3g}")

fig, axes = plt.subplots(3, 2, figsize=(8, 8.5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].set_title("Reconstruction\nSART")
ax[0].imshow(reconstruction_sart, cmap=plt.cm.Greys_r)

ax[1].set_title("Reconstruction error\nSART")
ax[1].imshow(reconstruction_sart - image, cmap=plt.cm.Greys_r, **imkwargs)

# Run a second iteration of SART by supplying the reconstruction
# from the first iteration as an initial estimate
reconstruction_sart2 = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart)
error = reconstruction_sart2 - image
print("SART (2 iterations) rms reconstruction error: "
      f"{np.sqrt(np.mean(error**2)):.3g}")

ax[2].set_title("Reconstruction\nSART, 2 iterations")
ax[2].imshow(reconstruction_sart2, cmap=plt.cm.Greys_r)

ax[3].set_title("Reconstruction error\nSART, 10 iterations")
ax[3].imshow(reconstruction_sart2 - image, cmap=plt.cm.Greys_r, **imkwargs)

reconstruction_sart10 = reconstruction_sart2
for _ in range(8):
    reconstruction_sart10 = iradon_sart(sinogram, theta=theta,
                                   image=reconstruction_sart10)
    
error = reconstruction_sart10 - image
print("SART (10 iterations) rms reconstruction error: "
      f"{np.sqrt(np.mean(error**2)):.3g}")
    
ax[4].set_title("Reconstruction\nSART, 10 iterations")
ax[4].imshow(reconstruction_sart10, cmap=plt.cm.Greys_r)

ax[5].set_title("Reconstruction error\nSART, 10 iterations")
ax[5].imshow(reconstruction_sart10 - image, cmap=plt.cm.Greys_r, **imkwargs)

fig.tight_layout()
plt.show()


# ***
# ### Вывод: изучили принципы построения томографических изображений.
