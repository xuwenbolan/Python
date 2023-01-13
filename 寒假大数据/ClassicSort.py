nums=[1.5,2.34,6.123,900,11,22,1,77,55,44444444]

def bubble_sort(list):
    for i in range(1,len(list)):
        for j in range(0,len(list)-i):
            if(list[j]>list[j+1]):
                temp=list[j]
                list[j]=list[j+1]
                list[j+1]=temp
    return list

def selection_sort(list):
    for i in range(0,len(list)-1):
        min_id=i
        for j in range(i+1,len(list)-1):
            if(list[j]<list[min_id]):
                min_id=j
        if i != min_id:
            list[i], list[min_id] = list[min_id], list[i]
    return list

def insertion_sort(list):
    for i in range(1,len(list)):
        key=list[i]
        j = i-1
        while j>=0 and key<list[j]:
            list[j+1]=list[j]
            j=j-1
        list[j+1]=key
    return list

# bubble_sort(nums)
# selection_sort(nums)
insertion_sort(nums)
for i in nums:
    print(i)