import semprop_pipeline as sp

semp = sp.init_test()
ms = semp.sem_prop_pipeline()

f = open("matchings.txt", "w+")
for m in ms:
    f.write(m)
f.close()

