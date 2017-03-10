import numpy as np

class AngleDerivation:

    # return the angle between two vectors in degrees
    @staticmethod
    def findAngle(vector1, vector2):
        x1, y1 = vector1
        x2, y2 = vector2
        
        dot = x1 * x2 + y1 * y2
        
        magnitude1 = np.sqrt(x1 * x1 + y1 * y1)
        magnitude2 = np.sqrt(x2 * x2 + y2 * y2)

        cosTheta = dot / (magnitude1 * magnitude2)
        radians = 0
        
        if(cosTheta >= 1):
            radians = 0
        elif(cosTheta <= -1):
            radians = np.pi
        else:
            radians = np.arccos(cosTheta)
            
        return np.degrees(radians)