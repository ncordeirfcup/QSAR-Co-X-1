import pandas as pd
import numpy as np
from statsmodels.multivariate.manova import MANOVA

class ycrandom():
      def __init__(self,df,nc,desc,ni):
            self.df=df
            self.nc=nc
            self.desc=desc
            self.ni=ni
      
      def shuffling(self,df, n=1, axis=0):     
          df = df.copy()
          for _ in range(n):
              df.apply(np.random.shuffle, axis=axis)
          return df
      
      def calculation(self,df,n,nf):
          ci=pd.DataFrame(df.iloc[:,n])
          hi=ci.columns.values.tolist()
          dff=df[df.iloc[:,1]==1]
          dffs=dff.iloc[:,2:]
          dff2=dffs.groupby(hi).mean()
          dff4=pd.merge(df,dff2, on=hi, how='left',suffixes=('?', '!')).fillna(0)
          dff4.columns=dff4.columns.str.rstrip('?')
          dff4.columns=dff4.columns.str.rstrip('!')
          fc=nf+2
          dff5=dff4.iloc[:,fc:]
          x_,y_=dff5.shape
          a=y_*0.5
          a=int(a)
          ldf=dff5.iloc[:,0:a]
          rdf=dff5.iloc[:,a:]
          li=[]
          for j in ldf:
              x=ldf[j]-rdf[j]
              li.append(x)
              trd=pd.DataFrame(li)
          return trd
      
      def boxjenk(self,df,nc):
          lt=[]
          for i in range(2,nc+2):
              li=pd.DataFrame(self.calculation(df,i,nc)).transpose().add_suffix('_'+pd.DataFrame(df.iloc[:,i]).columns.tolist()[0])
              lt.append(li)
              ad=pd.concat(lt,axis=1, join='outer')
              ad=pd.concat([df.iloc[:,0],df.iloc[:,1],ad],axis=1)
          return ad
      
      def randomization(self):
          C=[]
          for i in range(1,self.ni):
              yr=self.shuffling(self.df.iloc[:,0:2])
              c=self.nc
              cr=self.shuffling(self.df.iloc[:,2:c+2])
              xr=self.df.iloc[:,c+2:]
              ndr=pd.concat([yr,cr,xr],axis=1)
              dfbjr=self.boxjenk(ndr.iloc[:,0:-1],c)
              s=self.df.iloc[:,-1:]
              dfbjr=pd.concat([dfbjr,s],axis=1)
              dfbjtr=dfbjr[dfbjr['Set']=='Sub_train']
              xrd=dfbjtr[self.desc]
              yr=dfbjtr[yr.columns]
              table=MANOVA.from_formula('xrd.values~ yr.values',data=dfbjtr).mv_test().results['yr.values']['stat']
          C.append(np.mean(table.iloc[0,0])) 
          return C  