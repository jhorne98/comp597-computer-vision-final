import re
import sys
import csv
from itertools import zip_longest

def main():
    losses_txt = sys.argv[1]
    all_losses = []
    with open(losses_txt, 'r') as f:
        for line in f:
            line = line.split(" ", 1)
            losses = line[1]
            losses = re.split(r"[,]+", losses)
            losses = map(lambda s: s.replace(" ", ""), losses)
            losses = map(lambda s: s.replace("\n", ""), losses)
            losses = map(lambda s: s.replace("]", ""), losses)
            losses = list(map(lambda s: s.replace("[", ""), losses))
            losses.insert(0, line[0].replace(".pth:", ""))
            all_losses.append(losses)

    print(all_losses)
    export_data = zip_longest(*all_losses, fillvalue = '')
    with open(losses_txt.replace(".txt", ".csv"), 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(export_data)

if __name__ == '__main__':
    main()