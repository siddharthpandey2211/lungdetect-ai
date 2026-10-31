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

create table if not exists public.mri_analyses (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    result jsonb not null,
    image_path text,
    image_mime_type text not null
        check (image_mime_type in ('image/jpeg', 'image/png', 'image/webp')),
    image_size_bytes bigint not null
        check (image_size_bytes between 1 and 10485760),
    created_at timestamptz not null default now(),
    unique (id, user_id)
);

create index if not exists mri_analyses_user_created_idx
    on public.mri_analyses (user_id, created_at desc);

create table if not exists public.ct_scan_analyses (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references auth.users(id) on delete cascade,
    result jsonb not null,
    image_path text,
    image_mime_type text not null
        check (image_mime_type in ('image/jpeg', 'image/png', 'image/webp')),
    image_size_bytes bigint not null
        check (image_size_bytes between 1 and 10485760),
    created_at timestamptz not null default now(),
    unique (id, user_id)
);

create index if not exists ct_scan_analyses_user_created_idx
    on public.ct_scan_analyses (user_id, created_at desc);

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

alter table public.chat_messages
    alter column analysis_id drop not null;
alter table public.chat_messages
    add column if not exists mri_analysis_id uuid,
    add column if not exists ct_scan_analysis_id uuid;

alter table public.chat_messages
    drop constraint if exists chat_messages_mri_analysis_id_fkey;
alter table public.chat_messages
    add constraint chat_messages_mri_analysis_id_fkey
    foreign key (mri_analysis_id, user_id)
    references public.mri_analyses (id, user_id)
    on delete cascade;

alter table public.chat_messages
    drop constraint if exists chat_messages_ct_scan_analysis_id_fkey;
alter table public.chat_messages
    add constraint chat_messages_ct_scan_analysis_id_fkey
    foreign key (ct_scan_analysis_id, user_id)
    references public.ct_scan_analyses (id, user_id)
    on delete cascade;

alter table public.chat_messages
    drop constraint if exists chat_messages_exactly_one_analysis_check;
alter table public.chat_messages
    add constraint chat_messages_exactly_one_analysis_check
    check (num_nonnulls(analysis_id, mri_analysis_id, ct_scan_analysis_id) = 1);

create index if not exists chat_messages_analysis_created_idx
    on public.chat_messages (analysis_id, created_at);
create index if not exists chat_messages_mri_analysis_created_idx
    on public.chat_messages (mri_analysis_id, created_at);
create index if not exists chat_messages_ct_analysis_created_idx
    on public.chat_messages (ct_scan_analysis_id, created_at);

create table if not exists public.postal_codes (
    pin_code text primary key check (pin_code ~ '^[1-9][0-9]{5}$'),
    area_name text not null,
    district text not null,
    state text not null default 'Maharashtra',
    latitude double precision not null check (latitude between -90 and 90),
    longitude double precision not null check (longitude between -180 and 180),
    coordinate_source text not null,
    updated_at timestamptz not null default now()
);

create table if not exists public.providers (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    specialty text not null,
    hospital_affiliation text,
    address text not null,
    city text not null,
    state text not null,
    postal_code text not null check (postal_code ~ '^[1-9][0-9]{5}$'),
    latitude double precision not null check (latitude between -90 and 90),
    longitude double precision not null check (longitude between -180 and 180),
    services text[] not null check (cardinality(services) > 0),
    phone text,
    appointment_url text,
    website_url text,
    coordinate_source text,
    verification_source text not null,
    last_verified_at date not null,
    verified boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- These columns are additive so existing provider records remain valid. Keep
-- the defaults conservative: missing clinical evidence must not be treated as
-- a documented capability.
alter table public.providers
    add column if not exists ownership_type text not null default 'unknown',
    add column if not exists lung_cancer_expertise boolean not null default false,
    add column if not exists multidisciplinary_care boolean not null default false,
    add column if not exists diagnostic_capabilities text[] not null default '{}',
    add column if not exists address_verified boolean not null default false,
    add column if not exists government_supported boolean not null default false,
    add column if not exists charitable_assistance boolean not null default false,
    add column if not exists cost_information text,
    add column if not exists website_url text,
    add column if not exists coordinate_source text;

alter table public.providers
    drop constraint if exists providers_ownership_type_check;
alter table public.providers
    add constraint providers_ownership_type_check
    check (ownership_type in ('government', 'private', 'charitable', 'unknown'));

create index if not exists providers_verified_idx
    on public.providers (verified);

alter table public.postal_codes enable row level security;
grant select on public.postal_codes to anon, authenticated;

drop policy if exists "Public users read Mumbai-region PIN coordinates" on public.postal_codes;
create policy "Public users read Mumbai-region PIN coordinates"
    on public.postal_codes for select to anon, authenticated
    using (true);

alter table public.providers enable row level security;
grant select on public.providers to anon, authenticated;

drop policy if exists "Public users read verified providers" on public.providers;
create policy "Public users read verified providers"
    on public.providers for select to anon, authenticated
    using (verified = true);

alter table public.analyses enable row level security;
alter table public.mri_analyses enable row level security;
alter table public.ct_scan_analyses enable row level security;
alter table public.chat_messages enable row level security;
grant usage on schema public to authenticated;
grant select, insert, update, delete on public.analyses, public.mri_analyses,
    public.ct_scan_analyses, public.chat_messages to authenticated;

drop policy if exists "Users manage their own analyses" on public.analyses;
create policy "Users manage their own analyses"
    on public.analyses for all to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users manage their own MRI analyses" on public.mri_analyses;
create policy "Users manage their own MRI analyses"
    on public.mri_analyses for all to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users manage their own CT analyses" on public.ct_scan_analyses;
create policy "Users manage their own CT analyses"
    on public.ct_scan_analyses for all to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

drop policy if exists "Users manage their own chat" on public.chat_messages;
create policy "Users manage their own chat"
    on public.chat_messages for all to authenticated
    using ((select auth.uid()) = user_id)
    with check (
        (select auth.uid()) = user_id
        and (
            exists (
                select 1 from public.analyses
                where analyses.id = chat_messages.analysis_id
                  and analyses.user_id = (select auth.uid())
            )
            or exists (
                select 1 from public.mri_analyses
                where mri_analyses.id = chat_messages.mri_analysis_id
                  and mri_analyses.user_id = (select auth.uid())
            )
            or exists (
                select 1 from public.ct_scan_analyses
                where ct_scan_analyses.id = chat_messages.ct_scan_analysis_id
                  and ct_scan_analyses.user_id = (select auth.uid())
            )
        )
    );

insert into storage.buckets (id, name, public)
values ('pathology-images', 'pathology-images', false)
on conflict (id) do update
set public = false,
    file_size_limit = 52428800,
    allowed_mime_types = array['image/jpeg', 'image/png', 'image/webp'];

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
    ('mri-images', 'mri-images', false, 10485760, array['image/jpeg', 'image/png', 'image/webp']),
    ('ct-scan-images', 'ct-scan-images', false, 10485760, array['image/jpeg', 'image/png', 'image/webp'])
on conflict (id) do update
set public = false,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

drop policy if exists "Users read their MRI images" on storage.objects;
create policy "Users read their MRI images"
    on storage.objects for select to authenticated
    using (
        bucket_id = 'mri-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users upload their MRI images" on storage.objects;
create policy "Users upload their MRI images"
    on storage.objects for insert to authenticated
    with check (
        bucket_id = 'mri-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users delete their MRI images" on storage.objects;
create policy "Users delete their MRI images"
    on storage.objects for delete to authenticated
    using (
        bucket_id = 'mri-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users read their CT images" on storage.objects;
create policy "Users read their CT images"
    on storage.objects for select to authenticated
    using (
        bucket_id = 'ct-scan-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users upload their CT images" on storage.objects;
create policy "Users upload their CT images"
    on storage.objects for insert to authenticated
    with check (
        bucket_id = 'ct-scan-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

drop policy if exists "Users delete their CT images" on storage.objects;
create policy "Users delete their CT images"
    on storage.objects for delete to authenticated
    using (
        bucket_id = 'ct-scan-images'
        and (storage.foldername(name))[1] = (select auth.uid())::text
    );

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