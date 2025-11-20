from dagster import ScheduleDefinition, DefaultScheduleStatus

from .jobs import job_lstm_full

schedule_weekday_0800 = ScheduleDefinition(
    job=job_lstm_full,
    cron_schedule='0 8 * * 1-5', # 08:00 Mon–Fri
    execution_timezone='Europe/Berlin',
    default_status=DefaultScheduleStatus.RUNNING
)
