# import sys
# class motionVector():
#     def __init__(self):
#         self.box=None
#         self.mv=None
#
# def __sumOfAbsoluteDifferences__(self,n,m,isInterpolated):
#         sum=0.0
#         current_picture=None
#         if isInterpolated:
#             current_picture=self.current_picture_interpolated
#         else:
#             current_picture=self.current_picture
#
#         for i in range(self.n):
#             for j in range(self.n):
#                 y_n, x_m = self.__position___(i+n,j+m)
#                 y, x = self.__position___(i,j)
#                 if x_m>=len(current_picture[0]) or x_m<0 or y_n>=len(current_picture) or  y_n<0:
#                     sum =sum + sys.float_info.max/10
#                 elif x>=len(current_picture[0]) or x<0 or y>=len(current_picture) or  y<0:
#                     sum =sum + sys.float_info.max/10
#                 else:
#
#                     sum = sum + (self.__makroBlock__(i,j,isInterpolated=isInterpolated)-self.__makroBlock__(i+n,j+m,isCurrent=False,isInterpolated=isInterpolated))*(self.__makroBlock__(i,j,isInterpolated=isInterpolated)-self.__makroBlock__(i+n,j+m,isCurrent=False,isInterpolated=isInterpolated))
#                     self.numOfcomparedMacroblocks=self.numOfcomparedMacroblocks+1
#         return sum
#
# def motionEstimation(ref_img,cur_img,box):
#     p=16
#     p_range=range(-p,p+1)
#     min_n=None
#     min_m=None
#     min_value=sys.float_info.max
#
#     for n in p_range:
#         for m in p_range:
#             value=self.__sumOfAbsoluteDifferences__(n,m,isInterpolated)
#             # print 'value[',n,',',m,']= ', value
#             if value<min_value:
#                 min_value=value
#                 min_n=n
#                 min_m=m
#     return [min_n,min_m]
def extractor(ImageBuffer,MotionVectorBuffer):
    frame_index=0
    while True:
        while frame_index not in ImageBuffer.keys():
            pass
        MotionVectorBuffer[frame_index]=None
        frame_index+=1
