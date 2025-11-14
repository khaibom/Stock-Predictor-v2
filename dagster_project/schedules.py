from dagster import ScheduleDefinition, DefaultScheduleStatus

from .jobs import job_weekday_0800

schedule_0800 = ScheduleDefinition(
    job=job_weekday_0800,
    cron_schedule='0 8 * * 1-5', # 08:00 Mon–Fri
    execution_timezone='Europe/Berlin',
    default_status=DefaultScheduleStatus.RUNNING
)