def get_campaign_time(value):
    if value in ['Month 1', 'Month 2']:
        return 'precampaign'
    elif value in ['Month 3']:
        return 'during_campaign'
    elif value in ['Month 4', 'Month 5', 'Month 6']:
        return 'postcampaign'