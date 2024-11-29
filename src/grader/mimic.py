import pandas as pd
from itertools import combinations
from src.grader.metrics import *


class OpenEndGrader:
    def __init__(self, y_pred, pred_proba, gender, race):
        """
        Initialize with predictions and group labels for gender and race.

        Parameters:
        - y_pred (list or pd.Series): Predictions for each sample.
        - gender (list or pd.Series): Gender group for each sample.
        - race (list or pd.Series): Race group for each sample.
        """
        self.y_pred = pd.Series(y_pred)
        self.pred_proba = pd.Series(pred_proba)
        self.gender = pd.Series(gender)
        self.race = pd.Series(race)

    def calculate_statistical_parity(self):
        """
        Calculate statistical parity for each group in `gender` and `race` and
        then calculate the parity difference between each pair of groups.

        Returns:
        - dict: Statistical parity values for each gender and race group comparison, 
                including individual group parity and pairwise parity differences.
        """
        # Results dictionary to store statistical parity values and differences
        parity_results = {'gender': {}, 'race': {}}

        # Calculate statistical parity for each gender group
        gender_groups = self.gender.unique()
        gender_parity = {}

        for group in gender_groups:
            # Filter predictions for the specific gender group
            filter_mask = (self.gender == group)
            sub_pred = self.y_pred[filter_mask]

            # Calculate statistical parity for the group
            parity = statistical_parity(sub_pred, filter_mask.astype(int))
            gender_parity[group] = parity

        # Store individual gender group performance
        parity_results['gender']['individual'] = gender_parity

        # Create a cross table for gender parity differences
        gender_diff_table = pd.DataFrame(
            index=gender_groups, columns=gender_groups)
        for group_1 in gender_groups:
            for group_2 in gender_groups:
                if group_1 != group_2:
                    parity_diff = abs(
                        gender_parity[group_1] - gender_parity[group_2])
                    gender_diff_table.loc[group_1, group_2] = parity_diff
                else:
                    # No difference with itself
                    gender_diff_table.loc[group_1, group_2] = 0
        parity_results['gender']['parity_difference_table'] = gender_diff_table

        # Calculate statistical parity for each race group
        race_groups = self.race.unique()
        race_parity = {}

        for group in race_groups:
            # Filter predictions for the specific race group
            filter_mask = (self.race == group)
            sub_pred = self.y_pred[filter_mask]

            # Calculate statistical parity for the group
            parity = statistical_parity(sub_pred, filter_mask.astype(int))
            race_parity[group] = parity

        # Store individual race group performance
        parity_results['race']['individual'] = race_parity

        # Create a cross table for race parity differences
        race_diff_table = pd.DataFrame(index=race_groups, columns=race_groups)
        for group_1 in race_groups:
            for group_2 in race_groups:
                if group_1 != group_2:
                    parity_diff = abs(
                        race_parity[group_1] - race_parity[group_2])
                    race_diff_table.loc[group_1, group_2] = parity_diff
                else:
                    # No difference with itself
                    race_diff_table.loc[group_1, group_2] = 0
        parity_results['race']['parity_difference_table'] = race_diff_table

        return parity_results

    def pairwise_t_test_groups(self):
        """
        Conducts two-tailed t-tests for each unique pair of race and gender groups.

        Returns:
        - pd.DataFrame: A DataFrame containing t-scores and p-values for each pair of race and gender groups.
        """
        # Results dictionary to store t-scores and p-values for each group pair
        t_test_results = {'Group Type': [], 'Group 1': [],
                          'Group 2': [], 't-score': [], 'p-value': []}

        # Calculate pairwise t-tests for each unique pair of race groups
        race_groups = self.race.unique()
        for group1, group2 in combinations(race_groups, 2):
            # Create a binary mask for the pair of groups (1 for group1, 0 for group2)
            group_mask = [
                1 if g == group1 else 0 for g in self.race if g in {group1, group2}]
            pred_masked = [p for p, g in zip(self.pred_proba, self.race) if g in {
                group1, group2}]

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
            group_mask = [
                1 if g == group1 else 0 for g in self.gender if g in {group1, group2}]
            pred_masked = [p for p, g in zip(self.pred_proba, self.gender) if g in {
                group1, group2}]

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
            'gender': manual_welch_anova(self.pred_proba, self.gender),
            'race': manual_welch_anova(self.pred_proba, self.race)
        }
        return results


class BinaryGrader:
    def __init__(self, y_pred, y_true, gender, race):
        """
        Initialize with predictions and group labels for gender and race.

        Parameters:
        - y_pred (list or pd.Series): Predictions for each sample.
        - gender (list or pd.Series): Gender group for each sample.
        - race (list or pd.Series): Race group for each sample.
        """
        self.y_pred = pd.Series(y_pred)
        self.y_true = pd.Series(y_true)#.map({True: 1, False: 0})
        self.gender = pd.Series(gender)
        self.race = pd.Series(race)

    def calculate_statistical_parity(self):
        """
        Calculate statistical parity for each group in `gender` and `race` and
        then calculate the parity difference between each pair of groups.

        Returns:
        - dict: Statistical parity values for each gender and race group comparison, 
                including individual group parity and pairwise parity differences.
        """
        # Results dictionary to store statistical parity values and differences
        parity_results = {'gender': {}, 'race': {}}

        # Calculate statistical parity for each gender group
        gender_groups = self.gender.unique()
        gender_parity = {}

        for group in gender_groups:
            # Filter predictions for the specific gender group
            filter_mask = (self.gender == group)
            sub_pred = self.y_pred[filter_mask]

            # Calculate statistical parity for the group
            parity = statistical_parity(sub_pred, filter_mask.astype(int))
            gender_parity[group] = parity

        # Store individual gender group performance
        parity_results['gender']['individual'] = gender_parity

        # Create a cross table for gender parity differences
        gender_diff_table = pd.DataFrame(
            index=gender_groups, columns=gender_groups)
        for group_1 in gender_groups:
            for group_2 in gender_groups:
                if group_1 != group_2:
                    parity_diff = abs(
                        gender_parity[group_1] - gender_parity[group_2])
                    gender_diff_table.loc[group_1, group_2] = parity_diff
                else:
                    # No difference with itself
                    gender_diff_table.loc[group_1, group_2] = 0
        parity_results['gender']['parity_difference_table'] = gender_diff_table

        # Calculate statistical parity for each race group
        race_groups = self.race.unique()
        race_parity = {}

        for group in race_groups:
            # Filter predictions for the specific race group
            filter_mask = (self.race == group)
            sub_pred = self.y_pred[filter_mask]

            # Calculate statistical parity for the group
            parity = statistical_parity(sub_pred, filter_mask.astype(int))
            race_parity[group] = parity

        # Store individual race group performance
        parity_results['race']['individual'] = race_parity

        # Create a cross table for race parity differences
        race_diff_table = pd.DataFrame(index=race_groups, columns=race_groups)
        for group_1 in race_groups:
            for group_2 in race_groups:
                if group_1 != group_2:
                    parity_diff = abs(
                        race_parity[group_1] - race_parity[group_2])
                    race_diff_table.loc[group_1, group_2] = parity_diff
                else:
                    # No difference with itself
                    race_diff_table.loc[group_1, group_2] = 0
        parity_results['race']['parity_difference_table'] = race_diff_table

        return parity_results

    def calculate_equal_opportunity(self):
        """Calculate equal opportunity for each gender and race group."""
        eo_results = {'gender': {}, 'race': {}}

        # Gender-based equal opportunity
        gender_groups = self.gender.unique()
        gender_eo = {}

        for group in gender_groups:
            filter_mask = (self.gender == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            eo = equal_opportunity(sub_pred, sub_true, filter_mask)
            gender_eo[group] = eo

        eo_results['gender']['individual'] = gender_eo

        gender_diff_table = pd.DataFrame(index=gender_groups, columns=gender_groups)
        for group_1 in gender_groups:
            for group_2 in gender_groups:
                if group_1 != group_2:
                    eo_diff = abs(gender_eo[group_1] - gender_eo[group_2])
                    gender_diff_table.loc[group_1, group_2] = eo_diff
                else:
                    gender_diff_table.loc[group_1, group_2] = 0
        eo_results['gender']['difference_table'] = gender_diff_table

        # Race-based equal opportunity
        race_groups = self.race.unique()
        race_eo = {}

        for group in race_groups:
            filter_mask = (self.race == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            eo = equal_opportunity(sub_pred, sub_true, filter_mask)
            race_eo[group] = eo

        eo_results['race']['individual'] = race_eo

        race_diff_table = pd.DataFrame(index=race_groups, columns=race_groups)
        for group_1 in race_groups:
            for group_2 in race_groups:
                if group_1 != group_2:
                    eo_diff = abs(race_eo[group_1] - race_eo[group_2])
                    race_diff_table.loc[group_1, group_2] = eo_diff
                else:
                    race_diff_table.loc[group_1, group_2] = 0
        eo_results['race']['difference_table'] = race_diff_table

        return eo_results

    def calculate_equalized_odds(self):
        """Calculate equalized odds for each gender and race group."""
        eo_results = {'gender': {}, 'race': {}}

        # Gender-based equalized odds
        gender_groups = self.gender.unique()
        gender_eo = {}

        for group in gender_groups:
            filter_mask = (self.gender == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            eo = equalized_odds(sub_pred, sub_true, filter_mask)
            gender_eo[group] = eo

        eo_results['gender']['individual'] = gender_eo

        gender_diff_table = pd.DataFrame(index=gender_groups, columns=gender_groups)
        for group_1 in gender_groups:
            for group_2 in gender_groups:
                if group_1 != group_2:
                    eo_diff = abs(gender_eo[group_1] - gender_eo[group_2])
                    gender_diff_table.loc[group_1, group_2] = eo_diff
                else:
                    gender_diff_table.loc[group_1, group_2] = 0
        eo_results['gender']['difference_table'] = gender_diff_table

        # Race-based equalized odds
        race_groups = self.race.unique()
        race_eo = {}

        for group in race_groups:
            filter_mask = (self.race == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            eo = equalized_odds(sub_pred, sub_true, filter_mask)
            race_eo[group] = eo

        eo_results['race']['individual'] = race_eo

        race_diff_table = pd.DataFrame(index=race_groups, columns=race_groups)
        for group_1 in race_groups:
            for group_2 in race_groups:
                if group_1 != group_2:
                    eo_diff = abs(race_eo[group_1] - race_eo[group_2])
                    race_diff_table.loc[group_1, group_2] = eo_diff
                else:
                    race_diff_table.loc[group_1, group_2] = 0
        eo_results['race']['difference_table'] = race_diff_table

        return eo_results

    def calculate_overall_accuracy_equality(self):
        """Calculate overall accuracy equality for each gender and race group."""
        oae_results = {'gender': {}, 'race': {}}

        # Gender-based overall accuracy equality
        gender_groups = self.gender.unique()
        gender_oae = {}

        for group in gender_groups:
            filter_mask = (self.gender == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            oae = overall_accuracy_equality(sub_pred, sub_true, filter_mask)
            gender_oae[group] = oae

        oae_results['gender']['individual'] = gender_oae

        gender_diff_table = pd.DataFrame(index=gender_groups, columns=gender_groups)
        for group_1 in gender_groups:
            for group_2 in gender_groups:
                if group_1 != group_2:
                    oae_diff = abs(gender_oae[group_1] - gender_oae[group_2])
                    gender_diff_table.loc[group_1, group_2] = oae_diff
                else:
                    gender_diff_table.loc[group_1, group_2] = 0
        oae_results['gender']['difference_table'] = gender_diff_table

        # Race-based overall accuracy equality
        race_groups = self.race.unique()
        race_oae = {}

        for group in race_groups:
            filter_mask = (self.race == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            oae = overall_accuracy_equality(sub_pred, sub_true, filter_mask)
            race_oae[group] = oae

        oae_results['race']['individual'] = race_oae

        race_diff_table = pd.DataFrame(index=race_groups, columns=race_groups)
        for group_1 in race_groups:
            for group_2 in race_groups:
                if group_1 != group_2:
                    oae_diff = abs(race_oae[group_1] - race_oae[group_2])
                    race_diff_table.loc[group_1, group_2] = oae_diff
                else:
                    race_diff_table.loc[group_1, group_2] = 0
        oae_results['race']['difference_table'] = race_diff_table

        return oae_results

    def calculate_treatment_equality(self):
        """Calculate treatment equality for each gender and race group."""
        te_results = {'gender': {}, 'race': {}}

        # Gender-based treatment equality
        gender_groups = self.gender.unique()
        gender_te = {}

        for group in gender_groups:
            filter_mask = (self.gender == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            te = treatment_equality(sub_pred, sub_true, filter_mask)
            gender_te[group] = te

        te_results['gender']['individual'] = gender_te

        gender_diff_table = pd.DataFrame(index=gender_groups, columns=gender_groups)
        for group_1 in gender_groups:
            for group_2 in gender_groups:
                if group_1 != group_2:
                    te_diff = abs(gender_te[group_1] - gender_te[group_2])
                    gender_diff_table.loc[group_1, group_2] = te_diff
                else:
                    gender_diff_table.loc[group_1, group_2] = 0
        te_results['gender']['difference_table'] = gender_diff_table

        # Race-based treatment equality
        race_groups = self.race.unique()
        race_te = {}

        for group in race_groups:
            filter_mask = (self.race == group)
            sub_pred = self.y_pred[filter_mask]
            sub_true = self.y_true[filter_mask]

            te = treatment_equality(sub_pred, sub_true, filter_mask)
            race_te[group] = te

        te_results['race']['individual'] = race_te

        race_diff_table = pd.DataFrame(index=race_groups, columns=race_groups)
        for group_1 in race_groups:
            for group_2 in race_groups:
                if group_1 != group_2:
                    te_diff = abs(race_te[group_1] - race_te[group_2])
                    race_diff_table.loc[group_1, group_2] = te_diff
                else:
                    race_diff_table.loc[group_1, group_2] = 0
        te_results['race']['difference_table'] = race_diff_table

        return te_results
