#!usr/bin/python

def getFrac(nom, denom):
    frac = 'NaN'
    if denom == 'NaN':
        return frac
    elif denom == 0:
        return 0.
    else:
        if nom == 'NaN':
            return 0.
        else:
            frac = float(nom)/denom
    return frac

#end of file
