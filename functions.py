from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import kurtosis
from scipy.stats import skew

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import powerlaw
import statistics


def scaling_L1(df):
    indicator_L1AD = []
    alpha_L1AD = []
    Y0_L1AD = []
    residuos_L1AD = []
    bL = []
    bU = []
    IC = []

    for i in range(0,len(names)):   # len(names)
        for j in range(len(Codebook.Measure)):
            if (names[i] == Codebook.iloc[j,1]):      
                print(names[i])
                print(Codebook.iloc[j,6])
                df = pd.DataFrame({'x': np.log10(L1AD.BECTPOPL1AD)})
                df["y"] = np.log10(L1AD[names[i]])
                df["Category"] = L1AD.ISO2
                df = df.reset_index(drop=True)
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df[pd.notnull(df['y'])]

                y = df.y
                x = df.x
                print(len(x))
                model = LinearRegression()
                x = np.array(x)
                x = x.reshape(-1,1)
                model.fit(x,y)
                predd = model.predict(x)

                indicator_L1AD.append(Codebook.iloc[j,6])
                alpha_L1AD.append(round(model.coef_[0],3))
                Y0_L1AD.append(round(model.intercept_,3))
                residuos_L1AD.append(round(model.score(x, y),3))


                #if (round(popt[1],3) < 1.3) and (round(popt[1],3) > 0.1):
                R2 = 1-(1-model.score(x, y))*((len(df)-1)/(len(df)-2))

                X = sm.add_constant(x)
                mod = sm.OLS(y,X)
                res = mod.fit()
                ic = res.conf_int(0.1)
                ic2 = round(ic.iloc[1,0],3), round(ic.iloc[1,1],3)
                IC.append(ic2)

                res = y - predd

                plt.figure(figsize=(16, 9))

                plt.plot(x, predd, 'r-', linewidth=7.0, label = r'$\beta$ = {}'.format(round(model.coef_[0],3)) + "\n" + 
                           r'$R^2$ = {}'.format(round(R2,3)))

                df3 = df[df.Category == 'Other Cities']
                groups2 = df3.groupby("Category")

                for name, group2 in groups2:
                    plt.plot(group2["x"], group2["y"], marker="P", markersize = 15, linestyle="", label=name, 
                             color='orange')

                df2 = df[df.Category == 'BR']       
                groups = df2.groupby("Category")

                for name, group in groups:
                    plt.plot(group["x"], group["y"], marker="o", markersize = 15, linestyle="", label=name, 
                             color='g')

                #plt.title("L1AD", fontsize=35)
                plt.xlabel('log(Population)', fontsize=35)
                plt.ylabel('log({}'.format(Codebook.iloc[j,6])+')', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L1AD_Population_x_{}'.format(Codebook.iloc[j,6]))
                plt.show()

                ######################################

                plt.figure(figsize=(16, 9))
                np.var(res)

                plt.hist(res, bins=25, rwidth=0.9,label = 'Skew = {}'.format(round(skew(res),3)) + "\n" + 
                                               'Kurt = {}'.format(round(kurtosis(res),3)))                                               
                #width = 0.7 * (bins[1] - bins[0])
                #center = (bins[:-1] + bins[1:]) / 2
                #plt.bar(center, hist, align='center', width=width) 
                #plt.title('{}'.format(Codebook.iloc[j,6]), fontsize=35)
                plt.xlabel('Residuals', fontsize=35)
                plt.ylabel('Residuals count', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L1AD_Residuals_Population_x_{}'.format(Codebook.iloc[j,6]))
                plt.show()

                ######################################
                group1 = []
                group2 = []
                group3 = []
                group4 = []

                for l in range(len(res)):
                    if (L1AD.iloc[l,15] <= 250000):
                        group1.append(res.iloc[l])
                    elif (L1AD.iloc[l,15] > 250000) and (L1AD.iloc[l,15] <= 500000):
                        group2.append(res.iloc[l])
                    elif (L1AD.iloc[l,15] > 500000) and (L1AD.iloc[l,15] < 1000000):
                        group3.append(res.iloc[l])
                    else:
                        group4.append(res.iloc[l])

                plt.figure(figsize=(16, 9))
                bins = 25;
                plt.hist([group1,group2,group3, group4], bins, stacked=True, density=True, rwidth=0.9,
                          color=["red", "blue", "green", "orange"], 
                          label = ["< 0.25M", "0.25M to 0.5M", "0.5M to 1.0M", "> 1M"])

                plt.title('{}'.format(Codebook.iloc[j,6]), fontsize=35)
                plt.xlabel('Residuals', fontsize=35)
                plt.ylabel('Residuals count', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L1AD_GROUPS_Residuals_Population_x_{}'.format(Codebook.iloc[j,6]))
                
    data_L1AD = pd.DataFrame({'Y': indicator_L1AD, r'$\beta_{L1AD}$': alpha_L1AD, r'$95\%$ $CI_{L1AD}$': IC, r'$R^{2}_{L1AD}$': residuos_L1AD})
    data_L1AD.to_csv('/Users/aurelianosancho/Dropbox/Proposal_Aureliano/Results_proposal/Linear_L1AD.csv')
    
    return 'Done!'


def scaling_l1ux(df):
    
    indicator_L1UX = []
    alpha_L1UX = []
    Y0_L1UX = []
    residuos_L1UX = []
    IC2 = []

    for i in range(0,len(n_L1UX)):   # len(n_L1UX)
        for j in range(len(Codebook.Measure)):
            if (n_L1UX[i] == Codebook.iloc[j,2]):
                df = pd.DataFrame({'x': np.log10(L1UX.BECTPOPL1UX)})
                df["y"] = np.log10(L1UX[n_L1UX[i]])
                df["Category"] = L1UX.ISO2

                df = df.reset_index()
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df[pd.notnull(df['y'])]

                y = df.y
                x = df.x

                print(len(x))

                model = LinearRegression()
                x = np.array(x)
                x = x.reshape(-1,1)
                model.fit(x,y)
                predd = model.predict(x)

                indicator_L1UX.append(Codebook.iloc[j,6])
                alpha_L1UX.append(round(model.coef_[0],3))
                Y0_L1UX.append(round(model.intercept_,3))
                residuos_L1UX.append(round(model.score(x, y),3))

                #if (round(popt[1],3) < 1.3) and (round(popt[1],3) > 0.1):
                R2 = 1-(1-model.score(x, y))*((len(df)-1)/(len(df)-2))

                X = sm.add_constant(x)
                mod = sm.OLS(y,X)
                res = mod.fit()
                ic = res.conf_int(0.1)
                ic2 = round(ic.iloc[1,0],3), round(ic.iloc[1,1],3)
                IC2.append(ic2)

                res = y - predd

                plt.figure(figsize=(16, 9))
                plt.plot(x, predd, 'r-', linewidth=7.0, label = r'$\beta$ = {}'.format(round(model.coef_[0],3)) + "\n" + 
                           r'$R^2$ = {}'.format(round(R2,3)))

                df3 = df[df.Category == 'Other Cities']
                groups2 = df3.groupby("Category")

                for name, group2 in groups2:
                    plt.plot(group2["x"], group2["y"], marker="P", markersize = 15, linestyle="", label=name, 
                             color='orange')

                df2 = df[df.Category == 'BR']       
                groups = df2.groupby("Category")

                for name, group in groups:
                    plt.plot(group["x"], group["y"], marker="o", markersize = 15, linestyle="", label=name, 
                             color='g')

                #plt.title("L1UX", fontsize=35)
                plt.xlabel('log(Population)', fontsize=35)
                plt.ylabel('log({}'.format(Codebook.iloc[j,6])+')', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L1UX_Population_x_{}'.format(Codebook.iloc[j,6]))

                ######################################

                plt.figure(figsize=(16, 9))
                np.var(res)
                plt.hist(res, bins=25, rwidth=0.9,label = 'Skew = {}'.format(round(skew(res),3)) + "\n" + 
                                               'Kurt = {}'.format(round(kurtosis(res),3)))  
                plt.title('{}'.format(Codebook.iloc[j,6]), fontsize=35)
                plt.xlabel('Residuals', fontsize=35)
                plt.ylabel('Residuals count', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L1UX_Residuals_Population_x_{}'.format(Codebook.iloc[j,6]))
                plt.show()

                ######################################
                group1 = []
                group2 = []
                group3 = []
                group4 = []

                for l in range(len(res)):
                    if (L1UX.iloc[l,15] <= 250000):
                        group1.append(res.iloc[l])
                    elif (L1UX.iloc[l,15] > 250000) and (L1UX.iloc[l,15] <= 500000):
                        group2.append(res.iloc[l])
                    elif (L1UX.iloc[l,15] > 500000) and (L1UX.iloc[l,15] < 1000000):
                        group3.append(res.iloc[l])
                    else:
                        group4.append(res.iloc[l])

                plt.figure(figsize=(16, 9))
                bins = 25;
                plt.hist([group1,group2,group3, group4], bins, stacked=True, density=True, rwidth=0.9,
                          color=["red", "blue", "green", "orange"], 
                          label = ["< 0.25M", "0.25M to 0.5M", "0.5M to 1.0M", "> 1M"])

                plt.title('{}'.format(Codebook.iloc[j,6]), fontsize=35)
                plt.xlabel('Residuals', fontsize=35)
                plt.ylabel('Residuals count', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L1UX_GROUPS_Residuals_Population_x_{}'.format(Codebook.iloc[j,6]))

                
    data_L1UX = pd.DataFrame({'Y': indicator_L1UX, r'$\beta_{L1UX}$': alpha_L1UX, r'$95\%$ $CI_{L1UX}$': IC2, r'$R^{2}_{L1UX}$': residuos_L1UX})
    data_L1UX.to_csv('/Users/aurelianosancho/Dropbox/Proposal_Aureliano/Results_proposal/Linear_L1UX.csv')
                
    return 'Done!'


def scaling_l2(df):
    
    indicator_L2 = []
    alpha_L2 = []
    Y0_L2 = []
    residuos_L2 = []
    IC3 = []

    for i in range(0,len(n_L2)):   # len(n_L2)
        for j in range(len(Codebook.Measure)):
            if (n_L2[i] == Codebook.iloc[j,4]):
                df = pd.DataFrame({'x': np.log10(L2.BECTPOPL2)})
                df["y"] = np.log10(L2[n_L2[i]])
                df["Category"] = L2.ISO2

                df = df.reset_index()
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df[pd.notnull(df['y'])]

                y = df.y
                x = df.x

                model = LinearRegression()
                x = np.array(x)
                x = x.reshape(-1,1)
                model.fit(x,y)
                predd = model.predict(x)

                indicator_L2.append(Codebook.iloc[j,6])
                alpha_L2.append(round(model.coef_[0],3))
                Y0_L2.append(round(model.intercept_,3))
                residuos_L2.append(round(model.score(x, y),3))

                #if (round(popt[1],3) < 1.3) and (round(popt[1],3) > 0.1):
                R2 = 1-(1-model.score(x, y))*((len(df)-1)/(len(df)-2))

                X = sm.add_constant(x)
                mod = sm.OLS(y,X)
                res = mod.fit()
                ic = res.conf_int(0.1)
                ic2 = round(ic.iloc[1,0],3), round(ic.iloc[1,1],3)
                IC3.append(ic2)

                res = y - predd

                plt.figure(figsize=(16, 9))
                plt.plot(x, predd, 'r-', linewidth=7.0, label = r'$\beta$ = {}'.format(round(model.coef_[0],3)) + "\n" + 
                           r'$R^2$ = {}'.format(round(R2,3)))

                df3 = df[df.Category == 'Other Cities']
                groups2 = df3.groupby("Category")

                for name, group2 in groups2:
                    plt.plot(group2["x"], group2["y"], marker="P", markersize = 15, linestyle="", label=name, 
                             color='orange')

                df2 = df[df.Category == 'BR']       
                groups = df2.groupby("Category")

                for name, group in groups:
                    plt.plot(group["x"], group["y"], marker="o", markersize = 15, linestyle="", label=name, 
                             color='g')

                #plt.title("L2", fontsize=35)
                plt.xlabel('log(Population)', fontsize=35)
                plt.ylabel('log({}'.format(Codebook.iloc[j,6])+')', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L2_Population_x_{}'.format(Codebook.iloc[j,6]))

                ######################################

                plt.figure(figsize=(16, 9))
                np.var(res)
                plt.hist(res, bins=25, rwidth=0.9,label = 'Skew = {}'.format(round(skew(res),3)) + "\n" + 
                                               'Kurt = {}'.format(round(kurtosis(res),3))) 
                #width = 0.7 * (bins[1] - bins[0])
                #center = (bins[:-1] + bins[1:]) / 2
                #plt.bar(center, hist, align='center', width=width) 
                #plt.title('{}'.format(Codebook.iloc[j,6]), fontsize=35)
                plt.xlabel('Residuals', fontsize=35)
                plt.ylabel('Residuals count', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L2_Residuals_Population_x_{}'.format(Codebook.iloc[j,6]))

                ######################################
                group1 = []
                group2 = []
                group3 = []
                group4 = []

                for l in range(len(res)):
                    if (L2.iloc[l,16] <= 250000):
                        group1.append(res.iloc[l])
                    elif (L2.iloc[l,16] > 250000) and (L2.iloc[l,16] <= 500000):
                        group2.append(res.iloc[l])
                    elif (L2.iloc[l,16] > 500000) and (L2.iloc[l,16] < 1000000):
                        group3.append(res.iloc[l])
                    else:
                        group4.append(res.iloc[l])

                plt.figure(figsize=(16, 9))
                bins = 25;
                plt.hist([group1,group2,group3, group4], bins, stacked=True, density=True, rwidth=0.9,
                          color=["red", "blue", "green", "orange"], 
                          label = ["< 0.25M", "0.25M to 0.5M", "0.5M to 1.0M", "> 1M"])

                #plt.title('{}'.format(Codebook.iloc[j,6]), fontsize=35)
                plt.xlabel('Residuals', fontsize=35)
                plt.ylabel('Residuals count', fontsize=35)
                plt.xticks(fontsize=35)
                plt.yticks(fontsize=35)
                plt.legend(fontsize=25)
                plt.savefig('.../L2_GROUPS_Residuals_Population_x_{}'.format(Codebook.iloc[j,6]))
                
    data_L2 = pd.DataFrame({'Y': indicator_L2, r'$\beta_{L2}$': alpha_L2, r'$95\%$ $CI_{L2}$': IC3, r'$R^{2}_{L2}$': residuos_L2})
    data_L2.to_csv('/Users/aurelianosancho/Dropbox/Proposal_Aureliano/Results_proposal/Linear_L2.csv')
    
    return 'Done!'
