create table if not exists public.analyses (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    kind text not null check (kind in ('risk', 'histopathology')),
    result jsonb not null,
    patient_answers jsonb,
    image_path text,
    created_at timestamptz not null default now(),
    unique (id, user_id)
);

create index if not exists analyses_user_created_idx
    on public.analyses (user_id, created_at desc);

create table if not exists public.chat_messages (
    id uuid primary key default gen_random_uuid(),
    analysis_id uuid not null,
    user_id uuid not null references auth.users(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null check (char_length(content) between 1 and 5000),
    created_at timestamptz not null default now(),
    foreign key (analysis_id, user_id)
        references public.analyses (id, user_id)
        on delete cascade
);

create index if not exists chat_messages_analysis_created_idx
    on public.chat_messages (analysis_id, created_at);

alter table public.analyses enable row level security;
alter table public.chat_messages enable row level security;
grant usage on schema public to authenticated;
grant select, insert, update, delete on public.analyses to authenticated;
grant select, insert, update, delete on public.chat_messages to authenticated;

drop policy if exists "Users manage their own analyses" on public.analyses;
create policy "Users manage their own analyses"
    on public.analyses for all to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users manage their own chat" on public.chat_messages;
create policy "Users manage their own chat"
    on public.chat_messages for all to authenticated
    using ((select auth.uid()) = user_id)
    with check (
        (select auth.uid()) = user_id
        and exists (
            select 1
            from public.analyses
            where analyses.id = chat_messages.analysis_id
              and analyses.user_id = (select auth.uid())
        )
    );

insert into storage.buckets (id, name, public)
values ('pathology-images', 'pathology-images', false)
on conflict (id) do update
set public = false,
    file_size_limit = 52428800,
    allowed_mime_types = array['image/jpeg', 'image/png', 'image/webp'];

drop policy if exists "Users read their pathology images" on storage.objects;
create policy "Users read their pathology images"
    on storage.objects for select to authenticated
    using (
        bucket_id = 'pathology-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users upload their pathology images" on storage.objects;
create policy "Users upload their pathology images"
    on storage.objects for insert to authenticated
    with check (
        bucket_id = 'pathology-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users delete their pathology images" on storage.objects;
create policy "Users delete their pathology images"
    on storage.objects for delete to authenticated
    using (
        bucket_id = 'pathology-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

notify pgrst, 'reload schema';