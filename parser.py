import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()

digits = 5

train_error = []
validation_error = []

for line in lines:
    if "train error" in line:
        pos = line.find("train error")
        offset = len("train error") + 1
        train_error.append(float(line[pos+offset:pos+offset+digits]))
    if "valid error" in line:
        pos = line.find("valid error")
        offset = len("valid error") + 1
        validation_error.append(float(line[pos+offset:pos+offset+digits]))



import matplotlib.pyplot as plt


te = train_error
ve = validation_error

output_name = sys.argv[1] + ".png"

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlabel("Epochs")
ax.set_ylabel("Error")
#plt.yscale('log')
plt.xticks(range(21),range(0,105,5))
ax.plot(te, label="Training error")
ax.plot(ve, label="Validation error")
lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
fig.savefig(output_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
