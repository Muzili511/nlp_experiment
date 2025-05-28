# def findNumbers(nums):
#     count = 0
#     for item in nums:
#         flag = True
#         while item > 0:
#             temp = item % 100
#             if temp < 10:
#                 flag = False
#                 break
#             item = item / 100
#         if flag:
#             count += 1
#     return count
#
# findNumbers([7894])
# def maximum69Number(num: int) -> int:
#     k = 1
#     while k < num:
#         k *= 10
#     while k // 10 > 0:
#         temp = (num // k) % 10
#         if (num / k) % 10 == 6:
#             num += 3 * k
#             break
#         k //= 10
#     return num
#
# maximum69Number(9969)
def sortColors(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    temp = [0] * 3
    for item in nums:
        temp[item] += 1
    k = 0
    for i, item in enumerate(temp):
        for j in range(item):
            nums[k] = i
            k += 1
    return

sortColors([2,0,2,1,1,0])