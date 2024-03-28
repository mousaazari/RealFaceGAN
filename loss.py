import matplotlib.pyplot as plt
import numpy as np

with open("nohup.out") as f:
    lines = f.readlines()
    dis_loss = []
    gen_loss = []
    for line in lines:
        tokens = line.split()
        dis_loss.append(tokens[1][:-1])
        gen_loss.append(tokens[3][:-1])

dis_loss = np.array(dis_loss).astype(np.float)
gen_loss = np.array(gen_loss).astype(np.float)

plt.plot(dis_loss)
plt.plot(gen_loss)
plt.title('GAN loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['discriminator', 'generator'], loc='upper left')
plt.savefig("loss.png")
plt.clf()
