def log_df(df, context, name):
    context.log.info(name)
    context.log.info(df.describe())
    context.log.info(df.head())
    context.log.info(df.tail())