# driver class to run the recommender

import numpy as np
import csv
from Tkinter import *

def getDeals():
    with open('test.csv') as dataset:
        readDataset = csv.reader(dataset, delimiter = ',')
        deals = []
        uids = []
        hotness_vals = []
        for row in readDataset:
            deal = row[0]
            uid = row[1]
            #hotness = row[2]
            deals.append(deal)
            uids.append(uid)
            #hotness_vals.append(hotness)
        print(deals)
        print(uids)
    return deals


def clicked():
    selectedDeal = textBar.get(1.0, END)
    print selectedDeal
    with open('similarity.csv') as sim:
        readSimilarity = csv.reader(sim, delimiter =',')
        for row in readSimilarity:
            for column in row:
                if (selectedDeal == column):
                    rec_deal = row[0]
                    rec_deals.append(rec_deal)
        print(rec_deals)
    recommendedDeals.insert(END, "Let's find you some deals!")
    return rec_deals

def getSelectedDeal(textBar):
    selectedDeal = textBar.get(1.0, END)



rec_deals = []
selectedDeal = ""
window = Tk()
window.title("Deal Recommender")
window.geometry('800x510')
window.configure(bg="#f2ffe6")
deals_data = getDeals()
deals_Line = ""
availableDeals = Label(window, text="Hot deals for today!")
availableDeals.pack()
dealsList = Text(window, height = 15, width = 100, bg= "#d9ffb3")
for i in range(13):
    deals_Line += '\n' + deals_data[i]
dealsList.insert(END, deals_Line)
dealsList.configure(state="disabled")
dealsList.pack()
textBar = Text(window, height = 2, width = 100)
textBar.pack()
search = Button(window, text="Search", command=clicked)
search.pack()
recom_deals = Label(window, text="Recommended deals for you!")
recom_deals.pack()
recommendedDeals = Text(window, height = 10, width = 100, bg="#ffffb3")
reco_deals_data = clicked()
reco_Line = ""
for i in range(0):
    reco_Line += '\n' + reco_deals_data[i]
recommendedDeals.insert(END, reco_Line)
recommendedDeals.configure(state="disabled")
recommendedDeals.pack()
window.mainloop()







