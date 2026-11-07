-- One-time migration for databases where interview_bookings already exists.
-- Review the duplicate-slot query first and resolve any rows it reports.
--
-- SELECT interview_date, interview_time, COUNT(*)
-- FROM interview_bookings
-- GROUP BY interview_date, interview_time
-- HAVING COUNT(*) > 1;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM interview_bookings
        GROUP BY interview_date, interview_time
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Duplicate interview slots exist. Resolve them before adding uq_interview_bookings_date_time.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_interview_bookings_date_time'
          AND conrelid = 'interview_bookings'::regclass
    ) THEN
        ALTER TABLE interview_bookings
        ADD CONSTRAINT uq_interview_bookings_date_time
        UNIQUE (interview_date, interview_time);
    END IF;
END
$$;
