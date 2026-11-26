-- Chert Internal Testing: reproduce the project setting applied on 2026-11-26 UTC.
-- Run explicitly against chert-facetime (krlbefcfgxcsnfhrjjof) using an authorized
-- administrative SQL connection. This is deployment configuration, not a schema
-- migration. It does not change the shared line's default for other projects.
begin;

do $$
declare
  target_project constant uuid := 'ec9d4f61-7735-441b-8c6a-2f66faa2cc41';
  target_line constant uuid := 'fa456eab-5f35-4a9d-b241-bdec84a3b1ed';
begin
  if not exists (
    select 1 from public.facetime_shared_line_settings
    where project_id = target_project and video_framing = 'fill'
  ) then
    perform public.facetime_update_customer_line(
      target_project, target_line, '{"video_framing":"fill"}'::jsonb
    );
  end if;

  if not exists (
    select 1 from public.facetime_shared_line_settings
    where project_id = target_project and video_framing = 'fill'
  ) then
    raise exception 'Chert Internal Testing Fill setting was not persisted';
  end if;
end;
$$;

select project_id, video_framing, updated_at
from public.facetime_shared_line_settings
where project_id = 'ec9d4f61-7735-441b-8c6a-2f66faa2cc41'::uuid;

commit;
