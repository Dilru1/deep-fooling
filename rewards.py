def cmg_reward(self, delta_heading_max=70, oob_cost=5):
        """
        Main goal :
            - Maximizes the running distance (or projected 2D boat speed) along a given direction
        Surrogate goal : 
            - Avoid catastrophic trajectories (oob_cost) 
                - keep angle from target route < delta_heading_max
        """  
        #Relevent obs
        cmg                    = self.observations['cmg']
        #cmg
        cmg_term = cmg
                        
        #Out-of bound term
        done             = False
        oob_cost_updated = 0
        if( abs(self.observations['heading_relative']) >= delta_heading_max):
            done             = True
            oob_cost_updated = oob_cost
            print(f"\n({self.name}) Reward out of bounds : ")
            print(f"      - heading/target heading = {self.observations['heading_relative']:.2f}°")
    
        reward =  cmg_term - oob_cost_updated 
       
        return reward, done  
