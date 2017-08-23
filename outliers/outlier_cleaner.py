#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = abs(net_worths - predictions)
    cleaned_data.append((ages[0][0], net_worths[0][0], errors[0][0]))

    for i in range(1,len(errors)):

        position = 0
        stored_data = []
        while position < len(cleaned_data) and errors[i][0] > cleaned_data[position][2]:
            stored_data.append(cleaned_data[position])
            position += 1

        stored_data.append((ages[i][0], net_worths[i][0], errors[i][0]))
        for j in range(position, len(cleaned_data)):
            stored_data.append(cleaned_data[j])

        cleaned_data = stored_data

    cleaned_data = cleaned_data[:(len(cleaned_data)*9/10)]

    return cleaned_data


