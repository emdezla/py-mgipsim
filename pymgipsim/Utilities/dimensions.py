def make_list_size_consistent(*lists):

    max_length = max([max(i) for i in lists])

    for i in range(len(lists)):
        lists[i] *= int(max_length - len(lists[i]))

    return lists