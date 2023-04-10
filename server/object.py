#bounding box of detecting results

class object:
    def __init__(self,x,y,w,h,index,c): #c is the type of the objects e.g. person is 1, car is 2
        self.x=x
        self.y=y
        self.w=w
        self.h=h
        self.index=index
        self.c=c
        self.distance=(0,0)
        self.feature_point_index=[]
        self.target_point_index=[]
        self.isMatched=False
        self.bestres=0
    def addfeaturepoint(self,x,y,index):
        if self.x<=x<=self.x+self.w and self.y<=y<=self.y+self.h:
            self.feature_point_index.append(index)

    def addtargetpoint(self,x,y,index):
        if self.x<=x<=self.x+self.w and self.y<=y<=self.y+self.h:
            self.target_point_index.append(index)

    def sameFeatureNumber(self,pointindex):
        count=0
        for i in self.feature_point_index:
            for j in pointindex:
                if i==j:
                    count=count+1
        return count


    def match(self,nextobjectlist):
        bestres=0
        bestindex=-1
        for i in range(len(nextobjectlist)):
            if nextobjectlist[i].c==self.c:
                res=self.sameFeatureNumber(nextobjectlist[i].target_point_index)
                if res>bestres:
                    bestres=res
                    bestindex=i
        if bestindex!=-1:
            if nextobjectlist[bestindex].isMatched:
                if bestres>nextobjectlist[bestindex].bestres:
                    distance=(abs(nextobjectlist[bestindex].x+nextobjectlist[bestindex].w/2-self.x+self.w/2),abs(nextobjectlist[bestindex].y+nextobjectlist[bestindex].h/2-self.y+self.h/2))
                    nextobjectlist[bestindex].index=self.index
                    nextobjectlist[bestindex].distance=distance
            else:
                nextobjectlist[bestindex].isMatched=True
                nextobjectlist[bestindex].index=self.index
                nextobjectlist[bestindex].bestres=bestres
                nextobjectlist[bestindex].distance=(abs(nextobjectlist[bestindex].x+nextobjectlist[bestindex].w/2-self.x+self.w/2),abs(nextobjectlist[bestindex].y+nextobjectlist[bestindex].h/2-self.y+self.h/2))


