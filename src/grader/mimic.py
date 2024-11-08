import pandas as pd
from itertools import combinations
from src.grader.metrics import statistical_parity, generic_t_test, manual_welch_anova

class OpenEndGrader:
    def __init__(self, y_pred, gender, race):
        """
        Initialize with predictions and group labels for gender and race.
        
        Parameters:
        - y_pred (list or pd.Series): Predictions for each sample.
        - gender (list or pd.Series): Gender group for each sample.
        - race (list or pd.Series): Race group for each sample.
        """
        self.y_pred = pd.Series(y_pred)
        self.gender = pd.Series(gender)
        self.race = pd.Series(race)

    def calculate_statistical_parity(self):
        """
        Calculate statistical parity for each pair of groups in `gender` and `race`.

        Returns:
        - dict: Statistical parity values for each gender and race group comparison.
        """
        # Results dictionary to store statistical parity values for each group
        parity_results = {'gender': {}, 'race': {}}
        
        # Calculate statistical parity between each pair of gender groups
        gender_groups = self.gender.unique()
        for i, group_1 in enumerate(gender_groups):
            for group_2 in gender_groups[i+1:]:
                # Create binary mask for group comparison
                mask = (self.gender == group_1).astype(int)  # 1 for group_1, 0 for other
                group_label = f"{group_1} vs {group_2}"
                
                # Calculate statistical parity
                parity = statistical_parity(self.y_pred, mask)
                parity_results['gender'][group_label] = parity
        
        # Calculate statistical parity between each pair of race groups
        race_groups = self.race.unique()
        for i, group_1 in enumerate(race_groups):
            for group_2 in race_groups[i+1:]:
                # Create binary mask for group comparison
                mask = (self.race == group_1).astype(int)  # 1 for group_1, 0 for other
                group_label = f"{group_1} vs {group_2}"
                
                # Calculate statistical parity
                parity = statistical_parity(self.y_pred, mask)
                parity_results['race'][group_label] = parity

        return parity_results
    
    def pairwise_t_test_groups(self):
        """
        Conducts two-tailed t-tests for each unique pair of race and gender groups.

        Returns:
        - pd.DataFrame: A DataFrame containing t-scores and p-values for each pair of race and gender groups.
        """
        # Results dictionary to store t-scores and p-values for each group pair
        t_test_results = {'Group Type': [], 'Group 1': [], 'Group 2': [], 't-score': [], 'p-value': []}

        # Calculate pairwise t-tests for each unique pair of race groups
        race_groups = self.race.unique()
        for group1, group2 in combinations(race_groups, 2):
            # Create a binary mask for the pair of groups (1 for group1, 0 for group2)
            group_mask = [1 if g == group1 else 0 for g in self.race if g in {group1, group2}]
            pred_masked = [p for p, g in zip(self.y_pred, self.race) if g in {group1, group2}]
            
            # Perform the t-test using the generic_t_test method
            t_score, p_value = generic_t_test(pred_masked, group_mask)
            
            # Store the result
            t_test_results['Group Type'].append('Race')
            t_test_results['Group 1'].append(group1)
            t_test_results['Group 2'].append(group2)
            t_test_results['t-score'].append(t_score)
            t_test_results['p-value'].append(p_value)
        
        # Calculate pairwise t-tests for each unique pair of gender groups
        gender_groups = self.gender.unique()
        for group1, group2 in combinations(gender_groups, 2):
            # Create a binary mask for the pair of groups (1 for group1, 0 for group2)
            group_mask = [1 if g == group1 else 0 for g in self.gender if g in {group1, group2}]
            pred_masked = [p for p, g in zip(self.y_pred, self.gender) if g in {group1, group2}]
            
            # Perform the t-test using the generic_t_test method
            t_score, p_value = generic_t_test(pred_masked, group_mask)
            
            # Store the result
            t_test_results['Group Type'].append('Gender')
            t_test_results['Group 1'].append(group1)
            t_test_results['Group 2'].append(group2)
            t_test_results['t-score'].append(t_score)
            t_test_results['p-value'].append(p_value)
        
        # Convert the results to a DataFrame for easy viewing
        return pd.DataFrame(t_test_results)
    
    def welch_anova_results(self):
        """
        Calculates Welch's ANOVA for both gender and race groups.

        Returns:
        - dict: Welch ANOVA results for gender and race groups.
        """
        results = {
            'gender': manual_welch_anova(self.y_pred, self.gender),
            'race': manual_welch_anova(self.y_pred, self.race)
        }
        return results
