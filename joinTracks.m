% Play with AVISO Meso-scale eddy product looking joining up eddies that
% have a 1 day gap in them, which causes the track number to change.
%
% Dec-2022, Pat Welch, pat@mousebrains.com

% day-of-month and calendar month that an eddy must exist on.
% the first value is the "start" date to measure duration from.

measurementMonthDOM = 0415; % Month/Day the eddy must exist on
targetMonthDOM = 0505; % Month/Day the eddy will be intercepted
durations = [60]; % Duration with respect to 5-May

latmin = 5; % South of Palau
latmax = 30; % North of ARCTERX area
lonmin = 116; % China
lonmax = 150; % East of Guam

latExtra = 10; % For extended lat dimensions
lonExtra = 10; % For extended lon dimensions

myDir = fileparts(mfilename("fullpath")); % Where this script is located
dataDir = fullfile(myDir, "data"); % Where the data is located
saveDir = fullfile(myDir, "data"); % Where to save matlab files to
figDir = fullfile(myDir, "figs"); % Where to write figures to

files = dir(fullfile(dataDir, "*.nc"));

if ~exist ("tbl", "var") || ~istable(tbl)
    ofn = fullfile(saveDir, "joined.mat");
    if exist(ofn, "file")
        tbl = load(ofn).tbl
    else
        tbl = table();
        for index = 1:numel(files)
            item = files(index);
            fn = fullfile(item.folder, item.name);
            qCyclonic = ~contains(fn, "nticyclon");
            fprintf("Reading %s %d\n", fn, qCyclonic);
            a = loadData(fn); % Load the entire dataset
            % Get all tracks that ever existed within the extended lat/lon box
            a = pruneBox(a, ...
                latmin-latExtra, latmax+latExtra, ...
                lonmin-lonExtra, lonmax+lonExtra);
            a = mergeTracks(a, measurementMonthDOM); % Merge tracks together if needed
            a = pruneBox(a, latmin, latmax, lonmin, lonmax); % Tighter box
            a.qCyclonic = qCyclonic + zeros(size(a.time));
            a.index = index + zeros(size(a.time));
            tbl = [tbl; a];
            clear a; % clean up after myself
        end % for files
        fprintf("Saving tbl, %d, to %s\n", size(tbl,1), ofn);
        save(ofn, "tbl", "-v7.3");
    end % if exist ofn
end % if tbl

tracks = mkTracks(tbl, measurementMonthDOM, targetMonthDOM, durations);
tracks

function a = mkTracks(tbl, measurementMonthDOM, targetMonthDOM, durations)
% Group by index and track
groupBy = ["index", "track"];
yr = unique(year(tbl.time));
mesDates = datetime(yr, floor(measurementMonthDOM/100), mod(measurementMonthDOM, 100));
tgtDates = datetime(yr, floor(targetMonthDOM/100), mod(targetMonthDOM, 100));

tracks = rowfun(@mkEndPoints, tbl, ...
    "InputVariables", ["time", "latitude", "longitude"], ...
    "GroupingVariables", groupBy, ...
    "OutputVariableNames", ["t0", "t1", "lat0", "lat1", "lon0", "lon1"] ...
    );

t0 = repmat(tracks.t0, [1, numel(mesDates)]);
t1 = repmat(tracks.t1, [1, numel(mesDates)]);
t = repmat(mesDates, [1, size(tracks, 1)])';
tracks = tracks(any(t0 <= t & t < t1,2),:); % mesDates are in range

t0 = repmat(tracks.t0, [1, numel(tgtDates)]);
t1 = repmat(tracks.t1, [1, numel(tgtDates)]);
t = repmat(tgtDates, [1, size(tracks, 1)])';
tracks = tracks(any(t0 <= t & t < t1,2),:); % tgtDates are in range

t0 = repmat(tracks.t0, [1, numel(tgtDates)]);
t1 = repmat(tracks.t1, [1, numel(tgtDates)]);
t = repmat(tgtDates + days(max(durations)), [1, size(tracks, 1)])';
tracks = tracks(any(t0 <= t & t < t1,2),:); % tgtDates+max duration are in range


tracks
error("GotMe");
names = string(tbl.Properties.VariableNames);
onames = [names(~ismember(names, groupBy)), "stime", "etime"];

a = rowfun(@(varargin) summarizeTrack(names, onames, monthDOM, durations, varargin{:}), ...
    tbl, ...
    "InputVariables", names, ...
    "GroupingVariables", groupBy, ...
    "SeparateInputs", true, ...
    "OutputVariableNames", onames, ...
    "OutputFormat", "table" ...
    );

a.postDays = days(a.etime - a.time);
a.preDays  = days(a.time - a.stime);
end % mkTracks

function varargout = summarizeTrack(names, onames, monthDOM, durations, varargin)
mon = floor(monthDOM / 100);
dom = mod(monthDOM, 100);

tbl = table();

for i = 1:numel(names)
    name = names(i);
    tbl.(name) = varargin{i};
end % for i

varargout = cell(size(onames));

indices = find(month(tbl.time) == mon & day(tbl.time) == dom);
if isempty(indices)
    warning("No matching times found for track %d", tbl.track(1));
    return
end % if
if numel(indices) > 1 % Spans a year
    indices = indices(1); % Take the first one
    warning("Track %d spans a year, %s to %s", tbl.track(1), tbl.time(1), tbl.time(end));
end % if sum

stime = tbl.time(indices);
btime = stime + days(min(durations));
etime = stime + days(max(durations));

if ~any(tbl.time <= btime), return; end % Starts too late
if ~any(tbl.time >= etime), return; end % Ends too early

qBefore = tbl.time == btime; % time to measure at

for i = 1:numel(onames)
    name = onames(i);
    if ~ismember(names, name), continue; end
    varargout{i} = tbl.(name)(qBefore);
end % for i

% Special fields
varargout{ismember(onames, "time")} = stime;
varargout{ismember(onames, "stime")} = min(tbl.time);
varargout{ismember(onames, "etime")} = max(tbl.time);
end % summarizeTracks

function [t0, t1, lat0, lat1, lon0, lon1] = mkEndPoints(time, lat, lon)
[t, ix] = unique(time); % Make sure they are sorted and unique, which they already should be
t0 = t(1);
t1 = t(end);
lat0 = lat(ix(1));
lat1 = lat(ix(end));
lon0 = lon(ix(1));
lon1 = lon(ix(end));
end % mkEndPoints

function tbl = mergeTracks(tbl, monthDOM)
% There are jumps in what appear to be nearly contiguous tracks.
% This causes the track numbers to change.
% So we're going to renumber the tracks to make them contiguous. 
% 
% N.B. Most of the examples I've invested appear to be 
%      typhon/tropical storm correlated.

gapLength = days(3); % Maximum gap length
maxDist = 100 * 1000; % Arbitrary maximum gap distance spacing in meters

dates = datetime( ...
    unique(union(year(tbl.time), year(tbl.time))), ...
    floor(monthDOM / 100), mod(monthDOM, 100));

for t = dates' % Walk through the target dates in chronilogical order
    for cnt = 1:10 % Limit myself to 10 tries to avoid an infinite loop
        tracks = rowfun(@mkEndPoints, tbl, ...
            "InputVariables", ["time", "latitude", "longitude"], ...
            "GroupingVariables", "track", ...
            "OutputVariableNames", ["t0", "t1", "lat0", "lat1", "lon0", "lon1"] ...
            );
        % Pull out tracks that existed near t
        tgtTracks = tracks(tracks.t0-gapLength <= t & tracks.t1+gapLength >= t,:);
        if isempty(tgtTracks), break; end % No tracks to examine

        % Now find candidate tracks to append to tgtTracks
        nFull = size(tracks,1);
        nTgt = size(tgtTracks,1);

        % Look at prepending to the tgt tracks
        t0 = repmat(tgtTracks.t0, [1,nFull])';
        t1 = repmat(tracks.t1, [1, nTgt]);
        dt = days(t0 - t1);
        [iRow, iCol] = ind2sub(size(t0), find(dt > 1 & dt <= gapLength));
        qBreak = true; % Break out of for cnt loop unless modified
        if ~isempty(iRow) % Some candidates to prepend
            dist = distance( ...
                tracks.lat1(iRow), tracks.lon1(iRow), ...
                tgtTracks.lat0(iCol), tgtTracks.lon0(iCol), ...
                wgs84Ellipsoid("meter"));
            q = dist <= maxDist; % Points which are reasonable close
            if any(q) % Some tracks which are close enough to prepend
                qBreak = false; % Look for more
                fprintf("Prepend %d for %s\n", sum(q), t)
                a = tracks(iRow(q),:);
                b = tgtTracks(iCol(q),:);
                for index = 1:size(a,1)
                    % Change track in tbl from a.track to b.track
                    tbl.track(tbl.track == a.track(index)) = b.track(index);
                end % for index
            end % if any
        end % iRow prepending

        % Look at appending to the tgt tracks
        t0 = repmat(tracks.t0, [1,nTgt]);
        t1 = repmat(tgtTracks.t1, [1, nFull])';
        dt = days(t0 - t1);
        [iRow, iCol] = ind2sub(size(t0), find(dt > 1 & dt <= gapLength));
        if ~isempty(iRow) % Some candidates to append
            dist = distance( ...
                tracks.lat1(iRow), tracks.lon1(iRow), ...
                tgtTracks.lat0(iCol), tgtTracks.lon0(iCol), ...
                wgs84Ellipsoid("meter"));
            q = dist <= maxDist; % Points which are reasonable close
            if any(q) % Some tracks which are close enough to prepend
                qBreak = false; % Look for more
                fprintf("Appending %d for %s\n", sum(q), t);
                a = tracks(iRow(q),:);
                b = tgtTracks(iCol(q),:);
                for index = 1:size(a,1)
                    % Change track in tbl from a.track to b.track
                    tbl.track(tbl.track == a.track(index)) = b.track(index);
                end % for index
            end % if any
        end % iRow
        if qBreak, break; end
    end % for cnt
    fprintf("Finished after %d iterations for %s\n", cnt, t);
end % for t
end % joinTracks

function tbl = loadData(fn)
a = osgl_get_netCDF(fn, ...
    "time", "track", "longitude", "latitude", ...
    "amplitude", "effective_area", "effective_contour_height", ...
    "effective_contour_shape_error", "effective_radius", ...
    "inner_contour_height", "speed_area", "speed_average", ...
    "speed_contour_height", "speed_contour_shape_error", ...
    "speed_radius");

tbl = table();
n = numel(a.time);
for name = string(fieldnames(a))'
    if name == "uniqueID", continue; end
    if numel(a.(name)) == n
        tbl.(name) = a.(name);
    end % if
end % for

% dt = seconds(datetime(1970,1,1) - datetime(1950,1,1));
% tbl.time = datetime(a.time - dt, "ConvertFrom", "posixtime");
tbl.month = month(tbl.time);
tbl.dom = day(tbl.time);
end % loadData

function a = pruneBox(tbl, latmin, latmax, lonmin, lonmax)
% Find tracks that existed in the lat/lon box
tracks = tbl.track(...
    tbl.latitude >= latmin & tbl.latitude <= latmax ...
    & tbl.longitude >= lonmin & tbl.longitude <= lonmax ...
    );
a = tbl(ismember(tbl.track, tracks),:); % All tracks that ever existed in lat/long box
end % pruneBox